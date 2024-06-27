from torchvision.transforms import v2

def preprocess(image, resize, device):
    # print(resize)
    trans = v2.Compose(
        [v2.ToPILImage(), v2.Resize(resize), v2.ToTensor(),]
    )
    return trans(image.cpu()).to(device)

def save_patched_pic(adv_image, path):
    transform = v2.Compose([v2.ToPILImage(mode="RGB"),])
    adv_image = transform(adv_image)
    adv_image.save(path, quality=100, sub_sampling=0)
