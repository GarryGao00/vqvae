import os

import einops
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms


def download_mnist():
    mnist = torchvision.datasets.MNIST(root='./data/mnist', download=True)
    # print('length of MNIST', len(mnist))
    # id = 10000
    # img, label = mnist[id]
    # print(img)
    # print(label)
    # img.save('work_dirs/tmp_mnist.jpg')
    # tensor = transforms.ToTensor()(img)
    # print(tensor.shape)
    # print(tensor.max())
    # print(tensor.min())


class MNISTImageDataset(Dataset):

    def __init__(self, img_shape=(28, 28)):
        super().__init__()
        self.img_shape = img_shape
        self.mnist = torchvision.datasets.MNIST(root='./data/mnist')

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index: int):
        img = self.mnist[index][0]
        pipeline = transforms.Compose(
            [transforms.Resize(self.img_shape),
             transforms.ToTensor()])
        return pipeline(img)

class CelebADataset(Dataset):

    def __init__(self, root, img_shape=(64, 64)):
        super().__init__()
        self.root = root
        self.img_shape = img_shape
        self.filenames = sorted(os.listdir(root))

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int):
        path = os.path.join(self.root, self.filenames[index])
        img = Image.open(path)
        pipeline = transforms.Compose([
            transforms.CenterCrop(168),
            transforms.Resize(self.img_shape),
            transforms.ToTensor()
        ])
        return pipeline(img)


class CIFAR10ImageDataset(Dataset):
    def __init__(self, img_shape=(32, 32)):
        super().__init__()
        self.img_shape = img_shape
        self.cifar10 = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=None)

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, index: int):
        img, _ = self.cifar10[index]
        pipeline = transforms.Compose([
            transforms.Resize(self.img_shape),
            transforms.ToTensor()
        ])
        return pipeline(img)

CELEBA_DIR = 'data/celebA/img_align_celeba'
CELEBA_HQ_DIR = 'data/celebA/celeba_hq_256'
def get_dataloader(type,
                   batch_size,
                   img_shape=None,
                   dist_train=False,
                   num_workers=4,
                   use_lmdb=False,
                   **kwargs):
    if type == 'MNIST':
        if img_shape is not None:
            dataset = MNISTImageDataset(img_shape)
        else:
            dataset = MNISTImageDataset()
    elif type == 'CIFAR10':
        if img_shape is not None:
            dataset = CIFAR10ImageDataset(img_shape)
        else:
            dataset = CIFAR10ImageDataset()
    elif type == 'CelebA':
        if img_shape is not None:
            kwargs['img_shape'] = img_shape
        dataset = CelebADataset(CELEBA_DIR, **kwargs)
    elif type == 'CelebAHQ':
        if img_shape is not None:
            kwargs['img_shape'] = img_shape
        dataset = CelebADataset(CELEBA_HQ_DIR, **kwargs)

    if dist_train:
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                sampler=sampler,
                                num_workers=num_workers)
        return dataloader, sampler
    else:
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers)
        return dataloader


if __name__ == '__main__':
    os.makedirs('work_dirs', exist_ok=True)
    download_mnist()

    if os.path.exists(CELEBA_HQ_DIR):
        dataloader = get_dataloader('CelebAHQ', 16)
        img = next(iter(dataloader))
        print(img.shape)
        N = img.shape[0]
        img = einops.rearrange(img,
                               '(n1 n2) c h w -> c (n1 h) (n2 w)',
                               n1=int(N ** 0.5))
        print(img.shape)
        print(img.max())
        print(img.min())
        img = transforms.ToPILImage()(img)
        img.save('work_dirs/tmp_celebahq.jpg')