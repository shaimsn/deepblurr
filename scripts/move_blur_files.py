import os
import random
import shutil


cur_fnames = os.listdir('./')
base_dir = '../../WF_final/'
train_dir = 'train_pics/'
val_dir = 'val_pics/'
test_dir = 'test_pics/'
final_dir = '../png_final'


def test_which_set(base, test_dir):
    fnames = os.listdir(test_dir)
    for f in fnames:
        if base in f:
        return True
    return False


# for f in fnames:
# 	if '.JPEG' not in f:
# 		continue
# 	base = '_'.join(f.split('_')[:-1])
# 	bnum = f.split('_')[-1].split('.')[0]
# 	if test_which_set(base, base_dir+train_fnames):
# 		shutil.copy2(f, base_dir+train_fnames+'orig_'+bnum+'_'+base+'.JPEG')
# 	elif test_which_set(base, base_dir+val_fnames):
# 		shutil.copy2(f, base_dir+val_fnames+'orig_'+bnum+'_'+base+'.JPEG')
# 	elif test_which_set(base, base_dir+test_fnames):
# 		shutil.copy2(f, base_dir+test_fnames+'orig_'+bnum+'_'+base+'.JPEG')
# 	else:
# 		print('no location found for: {}'.format(f))
# 		shutil.copy2(f, './not_found/'+f)


already_transferred = []
for f in cur_fnames:
    if '.png' not in f:
        continue
    base = '_'.join(f.split('_')[:-1])
    bnum = f.split('_')[-1].split('.')[0]
    if test_which_set(base, base_dir+train_dir):
        # shutil.copy2(base_dir+train_fnames+'orig_'+bnum+'_'+base+'.JPEG')
        shutil.copy2(f, final_dir+train_dir+'blur_'+bnum+'_'+base+'.png')
    elif test_which_set(base, base_dir+val_dir):
        shutil.copy2(f, final_dir + val_dir + 'blur_' + bnum + '_' + base + '.png')
    elif test_which_set(base, base_dir+test_dir):
        shutil.copy2(f, final_dir + test_dir + 'blur_' + bnum + '_' + base + '.png')
    else:
        print('no location found for: {}'.format(f))
        shutil.copy2(f, './not_found/'+f)
