import os
import random

base_dir = 'data/WF_final/'
paths = ['train_pics/', 'val_pics/', 'test_pics/']
percentage = .9
for folder in paths:
    fnames = os.listdir(base_dir+folder)
    blur_fnames = [os.path.join(folder, f) for f in fnames if f.startswith('blur')]
    blur_fnames = random.shuffle(blur_fnames)
    for b in blur_fnames[:int(percentage*len(blur_fnames))]:
        os.rename(base_dir+folder+b, base_dir+folder+'temp/'+b)