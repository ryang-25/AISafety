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

import numpy as np
import torch

from PIL import Image, ImageEnhance, ImageOps
from torchvision import transforms

from EvalBox.Defense.defense import Defense

class AugMix(Defense):
    def __init__(self,
                 model=None,
                 device=None,
                 optimizer=None,
                 scheduler=None,
                 **kwargs):
        super().__init__(model, device)
        self._parse(**kwargs)

        self.augmentations = [self.autocontrast, self.equalize, self.posterize,
                              self.rotate, self.solarize, self.shear_x,
                              self.shear_y, self.translate_x, self.translate_y]
    
    def _parse(self, **kwargs):
        """
        """
        # The severity of the operators. defaults to 3.
        self.severity = kwargs.get("severity", 3)
        self.alpha = kwargs.get("alpha", 1.)
        # The chosen dataset
        # self.dataset = kwargs.get("dataset", "CIFAR10")
        # if self.dataset == "CIFAR10":
        #     self.resize = kwargs.get("resize", 36)
        # elif self.dataset == "ImageNet":
        #     self.resize = kwargs.get("resize", 224)


    def int_parameter(self, n, max_level):
        """
        Scales the parameter to an integer in the range.
        """
        return int(n * max_level / 10)
    
    def float_parameter(self, n, max_level):
        return float(n) * max_level / 10

    def sample_level(self):
        return np.random.uniform(low=0.1, high=self.severity)

    def generate(self, train_loader=None, valid_loader=None, defense_enhanced_saver=None):
        

        pass

    def normalize(self, image):
        """Normalize input image channel-wise to zero mean and unit variance."""
        image = image.transpose(2, 0, 1)  # Switch to channel-first
        mean, std = np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])
        image = (image - mean[:, None, None]) / std[:, None, None]
        return image.transpose(1, 2, 0)

    def aug(self, images):
        """
        Take images and return augmented images.
        """
        self.model.eval()
        nat_images = images.cpu().numpy()
        mixed_images = []
        for image in nat_images:
            ws = np.float32(np.random.dirichlet([self.alpha] * self.width))
            m = np.float32(np.random.beta(self.aug_prob_coeff, self.aug_prob_coeff))
            mix = torch.zeros_like(image)
            for i in range(self.width):
                image_aug = image.copy()
                depth = self.depth # fix this later
                for _ in range(depth):
                    op = np.random.choice(self.augmentations)
                    image_aug = op(image_aug)
                # commutative
                mix += ws[i] * self.normalize(image_aug)
            mixed_images.append((1 - m) * self.normalize(image) + m * mix)

        return torch.from_numpy(mixed_images).to(self.device)

    def train(self, train_loader=None, epoch=None):
        for i, (image, label) in enumerate(train_loader):
            # Zero the gradient
            self.optimizer.zero_grad()

    # base augmentations below

    def autocontrast(self, pil_img):
        return ImageOps.autocontrast(pil_img)

    def equalize(self, pil_img):
         return ImageOps.equalize(pil_img)

    def posterize(self, pil_img):
        level = self.int_parameter(self.sample_level(), 4)
        return ImageOps.posterize(pil_img, 4 - level)

    def rotate(self, pil_img):
        """
        Magnitude of the rotation ranges from -30 to 30 degrees.
        """
        degrees = self.int_parameter(self.sample_level(), 30)
        if np.random.uniform() > 0.5:
            degrees = -degrees
        return pil_img.rotate(degrees, resample=Image.BILINEAR)

    def solarize(self, pil_img):
        level = self.int_parameter(self.sample_level(), 256)
        return ImageOps.solarize(pil_img, 256 - level)

    def shear_x(self, pil_img):
        level = self.int_parameter(self.sample_level(), 0.3)
        if np.random.uniform() > 0.5:
            level = -level
        return pil_img.transform((self.resize, self.resize),
                                Image.AFFINE, (1, level, 0, 0, 1, 0),
                                resample=Image.BILINEAR)


    def shear_y(self, pil_img):
        level = self.float_parameter(self.sample_level(), 0.3)
        if np.random.uniform() > 0.5:
            level = -level
        return pil_img.transform((self.resize, self.resize),
                                Image.AFFINE, (1, 0, 0, level, 1, 0),
                                resample=Image.BILINEAR)

    def translate_x(self, pil_img):
        level = self.int_parameter(self.sample_level(), self.resize / 3)
        if np.random.random() > 0.5:
            level = -level
        return pil_img.transform((self.resize, self.resize),
                                Image.AFFINE, (1, 0, level, 0, 1, 0),
                                resample=Image.BILINEAR)


    def translate_y(self, pil_img):
        level = self.int_parameter(self.sample_level(), self.resize / 3)
        if np.random.random() > 0.5:
            level = -level
        return pil_img.transform((self.resize, self.resize),
                                Image.AFFINE, (1, 0, 0, 0, 1, level),
                                resample=Image.BILINEAR)
