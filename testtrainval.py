import os
import random
import shutil
random.seed(7)
ls = os.listdir('castom-masks/img/')

path = 'castom-masks/img/'

savetrain = 'crosval/train/'
savetest = 'crosval/test/'
saveval = 'crosval/val/'

if os.path.isdir('crosval/'):
    shutil.rmtree('crosval/')
os.makedirs(savetest)
os.makedirs(savetrain)
os.makedirs(saveval)

print(len(ls))
ls.sort()
random.shuffle(ls)

for i,file in enumerate(ls):
    if i <= 30:
        shutil.copyfile(path+file,saveval+file)
    elif i <= 40:
        shutil.copyfile(path+file,savetest+file)
    else:
        shutil.copyfile(path+file,savetrain+file)