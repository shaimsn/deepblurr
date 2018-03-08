"""Split the SIGNS dataset into train/val/test and resize images to 64x64.

The SIGNS dataset comes into the following format:
    train_signs/
        0_IMG_5864.jpg
        ...
    test_signs/
        0_IMG_5942.jpg
        ...

Original images have size (3024, 3024).
Resizing to (64, 64) reduces the dataset size from 1.16 GB to 4.7 MB, and loading smaller images
makes training faster.

We already have a test set created, so we only need to split "train_signs" into train and val sets.
Because we don't have a lot of images and we want that the statistics on the val set be as
representative as possible, we'll take 20% of "train_signs" as val set.
"""

import argparse
import random
import os

from PIL import Image
from tqdm import tqdm
import numpy as np
import pdb

SIZE = 64

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/GOPRO_Large', help="Directory with the GOPRO dataset")
parser.add_argument('--output_dir', default='data/GOPRO_Dataset', help="Where to write the new data")


def resize_and_save(filename, output_dir, size=SIZE):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    filename_str_list = filename.split("/")
    ixs = [3,5,4]
    relevant_str = [filename_str_list[i] for i in ixs]
    temp = relevant_str[1]
    temp = temp[:-4]
    relevant_str[1] = temp
    temp = relevant_str[2]
    temp = temp + '.png'
    relevant_str[2] = temp
    to_save = '_'.join(relevant_str)
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)
    image.save(os.path.join(output_dir, to_save))


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Define the data directories
    train_data_dir = os.path.join(args.data_dir, 'train')
    test_data_dir = os.path.join(args.data_dir, 'test')

    # Get the foldernames inside GOPRO_Large/train
    train_foldernames = os.listdir(train_data_dir)
    train_blur_list = []
    train_sharp_list = []
    for foldername in train_foldernames:
        train_subfolder_dir = os.path.join(train_data_dir, foldername)

        subfolder_blur_dir = os.path.join(train_subfolder_dir, 'blur')
        train_blur_filenames = os.listdir(subfolder_blur_dir)
        train_blur_filenames = [os.path.join(subfolder_blur_dir, f) for f in train_blur_filenames if f.endswith('.png')]
        for name in train_blur_filenames:
            train_blur_list.append(name)

        subfolder_sharp_dir = os.path.join(train_subfolder_dir, 'sharp')
        train_sharp_filenames = os.listdir(subfolder_sharp_dir)
        train_sharp_filenames = [os.path.join(subfolder_sharp_dir, f) for f in train_sharp_filenames if f.endswith('.png')]
        for name in train_sharp_filenames:
            train_sharp_list.append(name)

    # Get the foldernames inside GOPRO_Large/test
    test_foldernames = os.listdir(test_data_dir)
    test_blur_list = []
    test_sharp_list = []
    for foldername in test_foldernames:
        test_subfolder_dir = os.path.join(test_data_dir, foldername)

        subfolder_blur_dir = os.path.join(test_subfolder_dir, 'blur')
        test_blur_filenames = os.listdir(subfolder_blur_dir)
        test_blur_filenames = [os.path.join(subfolder_blur_dir, f) for f in test_blur_filenames if f.endswith('.png')]
        for name in test_blur_filenames:
            test_blur_list.append(name)

        subfolder_sharp_dir = os.path.join(test_subfolder_dir, 'sharp')
        test_sharp_filenames = os.listdir(subfolder_sharp_dir)
        test_sharp_filenames = [os.path.join(subfolder_sharp_dir, f) for f in test_sharp_filenames if f.endswith('.png')]
        for name in test_sharp_filenames:
            test_sharp_list.append(name)

    # print("train_blur_list", train_blur_list)
    # print()
    # print("train_sharp_list", train_sharp_list)
    # print()
    # print("test_blur_list", test_blur_list)
    # print()
    # print("test_sharp_list", test_sharp_list)
    #so far, train_blur_filenames, train_sharp_filenames, test_blur_filenames and test_sharp_filenames lists created

    # Split the images in 'train_signs' into 80% train and 20% val

    # Shuffle with a fixed seed for reproducibility
    random.seed(230)
    # all_image_blur_names = train_blur_list + test_blur_list
    # all_image_sharp_names = train_sharp_list + test_sharp_list

    train_blur_len = len(train_blur_list)

    ixs_to_sort = np.linspace(0,train_blur_len-1,train_blur_len, dtype=int)
    random.shuffle(ixs_to_sort)
    train_len = train_blur_len*4//5

    train_filenames_list = []
    val_filenames_list = []
    for i in range(len(ixs_to_sort)):
        ix = ixs_to_sort[i]
        if (i <= train_len):
            train_filenames_list.append(train_blur_list[ix])
            train_filenames_list.append(train_sharp_list[ix])
        else:
            val_filenames_list.append(train_blur_list[ix])
            val_filenames_list.append(train_sharp_list[ix])


    filenames = {'train': train_filenames_list,
                 'val': val_filenames_list,
                 'test': test_blur_list+test_sharp_list}

    # pdb.set_trace()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    # Preprocess train, val, and test
    for split in ['train', 'val', 'test']:
        output_dir_split = os.path.join(args.output_dir, '{}_pics'.format(split))
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: dir {} already exists".format(output_dir_split))
        print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
        # pdb.set_trace()
        for filename in tqdm(filenames[split]):
            resize_and_save(filename, output_dir_split, size=SIZE)

    print("Done building dataset")
