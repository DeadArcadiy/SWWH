import os
import random
import shutil

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

for file in ls:
    a = int(random.random() * 100)
    if a % 4 == 0:
        if a % 3 != 0:
            shutil.copyfile(path+file,saveval+file)
        else:
            shutil.copyfile(path+file,savetest+file)
    else:
        shutil.copyfile(path+file,savetrain+file)