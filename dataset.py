import os

import einops
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

# Set this tp `True` and run this script to convert dataset to LMDB format
TO_LMDB = False

def download_mnist():
    mnist = torchvision.datasets.MNIST(root='./data/mnist', download=True)
    print('length of MNIST', len(mnist))
    id = 4
    img, label = mnist[id]
    print(img)
    print(label)

    # On computer with monitor
    # img.show()

    img.save('work_dirs/tmp_mnist.jpg')
    tensor = transforms.ToTensor()(img)
    print(tensor.shape)
    print(tensor.max())
    print(tensor.min())


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