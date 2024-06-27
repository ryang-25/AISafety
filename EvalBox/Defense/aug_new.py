# Copyright 2019 Google LLC
# Copyright 2024 Roland Yang
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
augmix.py

The set of augmentations from AugMix (really AutoAugment).

To summarize: we choose operations from the PIL library, taking care to not
pick operations that would potentially cause a misclassification by way of
manifold intrusion (1).

Rationale for each specific range of magnitudes was not alluded to, but they are
uniform and can be chosen from at random.

This set of operations is listed in the table below:

See Appendix C of https://arxiv.org/pdf/1805.09501.
"""

import os
import torch
from torchvision.transforms import v2

from utils.Defense_utils import adjust_learning_rate

from EvalBox.Defense.defense import Defense

class AugMix(Defense):
    def __init__(self, model=None, device=None, optimizer=None, scheduler=None, **kwargs):
        super().__init__(model, device)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self._parse_params(**kwargs)
    
    def _parse_params(self, **kwargs):
        self.dataset = str(kwargs.get("dataset"))
        self.num_epochs = int(kwargs.get("num_epochs", 100))

    def valid(self, valid_loader=None):
        """
        @description:
        @param {
            valid_loader:
            epoch:
        }
        @return: val_acc
        """
        device = self.device
        self.model.to(device).eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = self.model(inputs)
                preds = torch.argmax(outputs, 1)
                total += inputs.shape[0]
                correct += (preds == labels).sum().item()
            val_acc = correct / total
        return val_acc

    def train(self, train_loader=None, epoch=None):
        device = self.device

        transforms = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.AugMix(interpolation=v2.InterpolationMode.BILINEAR),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

        for i, (images, labels) in enumerate(train_loader):
            nat_images = images.to(device)
            nat_labels = labels.to(device)

            self.model.eval()
            aug_images = transforms(nat_images)

            logits_nat = self.model(nat_images)
            loss_nat = self.criterion(logits_nat, nat_labels)

            logits_aug = self.model(aug_images)
            loss_aug = self.criterion(logits_aug, nat_labels)

            loss = 0.5 * (loss_nat + loss_aug)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(
                '\rTrain Epoch {:>2}: [batch:{:>4}/{:>4}]  \tloss_nat={:.4f}, loss_aug={:.4f}, total_loss={:.4f} ===> '
                .format(epoch, i, len(train_loader), loss_nat.item(),
                        loss_aug.item(), loss.item()),
                end=' ')

    def generate(self, train_loader=None, valid_loader=None, defense_enhanced_saver=None):
        dir_path = os.path.dirname(defense_enhanced_saver)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        best_val_acc = None
        best_model_weight = self.model.state_dict()

        for epoch in range(self.num_epochs):
            self.train(train_loader, epoch)
            val_acc = self.valid(valid_loader)
            adjust_learning_rate(epoch=epoch, optimizer=self.optimizer)
            if not best_val_acc or round(val_acc, 4) >= round(best_val_acc, 4):
                if best_val_acc is not None:
                    os.remove(defense_enhanced_saver)
                best_val_acc = val_acc
                best_model_weights = self.model.state_dict()
                torch.save(self.model.state_dict(), defense_enhanced_saver)
            else:
                print('Train Epoch{:>3}: validation dataset accuracy did not improve from {:.4f}\n'.format(epoch,
                                                                                                           best_val_acc))
        print('Best val Acc: {:.4f}'.format(best_val_acc))
        return best_model_weights, best_val_acc
