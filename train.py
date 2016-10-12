import cv2
import tflearn
import os
import numpy as np
from tqdm import tqdm
import hickle
import random

def cache_data(data, path):
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'w')
        hickle.dump(data, file)
        file.close()
    else:
        print('Directory doesn\'t exists; Creating..')
        os.mkdir(os.path.dirname(path), 0755)
        file = open(path, 'w')
        hickle.dump(data, file)
        file.close()         
        
def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'r')
        data = hickle.load(file)
    return data

def load_imgs(rows, cols):
    progressBar = tqdm(total=15212)
    count = 0
    imgsPath = './center_images'
    imgFiles = os.listdir(imgsPath)
    imgs = np.ndarray((0, rows, cols, 3), dtype='float32')
    for imgFile in imgFiles:
        img = cv2.imread(os.path.join(imgsPath, imgFile))
        resizedImage = cv2.resize(img, (cols, rows), cv2.INTER_LINEAR)
        imgs = np.append(imgs, [resizedImage], axis=0)
        progressBar.update(1)
        count += 1
        if count % 500 == 0:
            cache_data(imgs, './cache/' + str(count/500) + '.dat')
            imgs = np.ndarray((0, rows, cols, 3), dtype='float32')
    cache_data(imgs, './cache/' + str(31) + '.dat')
    progressBar.close()
    return imgs

def load_gt():
    groundTruthFile = open('./groundTruth.csv')
    line = groundTruthFile.readline()
    groundTruth = []
    while line != '':
        arr = line.split(',')
        if len(arr) == 2:
            groundTruth.append(float(arr[1]))
        line = groundTruthFile.readline()
    return groundTruth

def loadDataFromCaches(masterXCaches, rows, cols):
    progressBar = tqdm(total=len(masterXCaches))
    cachePath = './cache'
    imgs = np.ndarray((0, rows, cols, 3), dtype='float32')
    for i in range(len(masterXCaches)):
        cacheFile = os.path.join(cachePath, str(masterXCaches[i]) + '.dat')
        imgs = np.append(imgs, restore_data(cacheFile), axis=0)
        progressBar.update(1)
    print imgs.shape
    progressBar.close()
    return imgs

def model():
    

def train():
    #masterX = load_imgs(224, 224) # Use this only if the input data is not cached
    numCaches = 31
    caches = range(31)
    caches = [c+1 for c in caches]
    random.shuffle(caches)
    masterXCaches = caches[:7]
    masterX = loadDataFromCaches(masterXCaches, 224, 224)
    masterY = load_gt()
    newY = []
    for i in range(len(masterXCaches)):
        pos = masterXCaches[i]
        start = (pos-1) * 500
        end = pos*500
        partY = masterY[start:end]
        newY.extend(partY)
    print len(newY)
    print 'Loaded data'


if __name__ == '__main__':
    train()
