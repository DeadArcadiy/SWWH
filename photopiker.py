import os
import random
import shutil

visited = {}

for file in os.listdir('castom-masks/img/'):
    visited[file] = 0

path = 'cropped/'
allimages = os.listdir(path)
destination = 'for_labeling/'

if not os.path.isfile(destination):
    shutil.rmtree(destination)
os.makedirs(destination)

counter = 0
for file in allimages:
    if int(random.random() * 100) % 5 == 0:
        if visited.get(file)!=None:
            print('opa')
        else:
            counter += 1
            shutil.copyfile(path+file,destination+file)
            print(path+file)


print(counter)


