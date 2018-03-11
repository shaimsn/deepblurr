import random
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pdb
import numpy as np

# borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# define a training image loader that specifies transforms on images. See documentation for more details.
# train_transformer = transforms.Compose([
#     # transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
#     transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
#     transforms.ToTensor()])  # transform it into a torch tensor
#
# # loader for evaluation, no horizontal flip
# eval_transformer = transforms.Compose([
#     transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
#     transforms.ToTensor()])  # transform it into a torch tensor


# define a training image loader that specifies transforms on images.
train_transformer = transforms.ToTensor()
# loader for evaluation
eval_transformer = transforms.ToTensor()


class GOPRODataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir, transform):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.filenames = os.listdir(data_dir)
        self.blur_filenames = [os.path.join(data_dir, f) for f in self.filenames if f.startswith('blur')]
        self.sharp_filenames = [os.path.join(data_dir, f) for f in self.filenames if f.startswith('sharp')]

        # self.labels = [int(os.path.split(filename)[-1][0]) for filename in self.filenames]
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.blur_filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image

                    base_fname = self.sharp_filenames[idx].split('/')[-1].split('.')[0]
        extension = self.sharp_filenames[idx].split('/')[-1].split('.')[1]
        path = self.sharp_filenames[idx].split('/')[:-2]
        path.append("kernels_dataset")
        path.append(base_fname+"_k1."+extension)
        blur_fname = "/".join(path)



        # input_image = Image.open(self.blur_filenames[idx])  # PIL image
        # input_image = self.transform(input_image)
        # label_image = Image.open(self.sharp_filenames[idx])  # PIL image
        # label_image = self.transform(label_image)
        """

        input_image = Image.open(self.blur_filenames[idx])
        input_image = np.array(input_image)
        input_image = np.reshape(input_image, (256, 256, 15, 3), order='F')
        # tested and this is actuall yhe correct version
        input_image = np.reshape(input_image, (256, 256, 45), order='C')
        input_image = self.transform(input_image)

        path = self.blur_filenames[idx].split('/')[:-1]
        # eliminate path and blur_prefix from fname
        base_name = '_'.join(self.blur_filenames[idx].split('/')[-1].split('_')[2:])
        # This is the actual filename appended to the directory structure
        path.append('sharp_' + base_name)
        label_image = Image.open('/'.join(path))  #because one sharp image for multiple training images
        label_image = np.array(label_image)
        label_image = label_image[:, :, :3] # 4th channel is transparency... cut it out
        label_image = self.transform(label_image)

        path = self.blur_filenames[idx].split('/')[:-1]
        output_name = '_'.join(self.blur_filenames[idx].split('/')[-1].split('_')[1:])
        path.append('out_'+output_name)

        return input_image, label_image, '/'.join(path)


def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}_pics".format(split))

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                my_dataset = GOPRODataset(path, train_transformer)
                # pdb.set_trace()
                dl = DataLoader(GOPRODataset(path, train_transformer), batch_size=params.train_batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
            else:
                dl = DataLoader(GOPRODataset(path, eval_transformer), batch_size=params.eval_batch_size, shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders
