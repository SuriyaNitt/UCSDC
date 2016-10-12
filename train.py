import cv2
import tflearn
import os
import numpy as np
from tqdm import tqdm
import hickle

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
    groundTruth.append(float(line.split(',')[1]))
    while line != '':
        line = groundTruthFile.readline()
        groundTruth.append(float(line.split(',')[1]))
    return groundTruth

def train():
    masterX = load_imgs(224, 224)
    masterY = load_gt()
    print 'Loaded data'


if __name__ == '__main__':
    train()
