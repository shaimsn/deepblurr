import os
import random

base_dir = 'data/png_final/'
paths = ['train_pics/', 'val_pics/', 'test_pics/']
percentage = .95
for folder in paths:
    fnames = os.listdir(base_dir+folder)
    # print(fnames)
    blur_fnames = [f for f in fnames if f.startswith('blur')]
    random.shuffle(blur_fnames)
    # print(blur_fnames)
    for b in blur_fnames[:int(percentage*len(blur_fnames))]:
        os.rename(base_dir+folder+b, base_dir+folder+'temp/'+b)
