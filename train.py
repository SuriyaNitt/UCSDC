import cv2
import tflearn
import os
import numpy as np
from tqdm import tqdm
import hickle
import random
from sklearn.cross_validation import train_test_split
import sys
import math
import re

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

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
    imgFiles.sort(key=natural_keys)
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

def network(rows, cols):
    # Building Residual Network
    net = tflearn.input_data(shape=[None, rows, cols, 3])
    net = tflearn.conv_2d(net, 32, 3, activation='relu', bias=False)
    net = tflearn.max_pool_2d(net, 2)
    # Residual blocks
    net = tflearn.residual_bottleneck(net, 2, 16, 64, downsample=True)
    net = tflearn.residual_bottleneck(net, 2, 32, 128, downsample=True)
    #net = tflearn.residual_bottleneck(net, 2, 64, 256, downsample=True)
    #net = tflearn.residual_bottleneck(net, 2, 128, 512, downsample=True)
    
    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'prelu')
    net = tflearn.global_avg_pool(net)
    
    # Regression
    net = tflearn.fully_connected(net, 512, activation='prelu')
    net = tflearn.fully_connected(net, 1, activation='prelu')
    net = tflearn.regression(net, optimizer='momentum',
                             loss='mean_square',
                             learning_rate=0.1)
    return net 

def train():
    #masterX = load_imgs(224, 224) # Use this only if the input data is not cached
    rows = 224
    cols = 224
    numCaches = 31
    trainedData = []#[10, 15, 6, 27, 24]
    caches = range(numCaches)
    caches = [c+1 for c in caches]
    random.shuffle(caches)
    for i in trainedData:
        caches.remove(i)
    masterXCaches = caches[5:10]
    print caches[5:10]
    masterX = loadDataFromCaches(masterXCaches, rows, cols)
    masterY = load_gt()
    newY = []
    for i in range(len(masterXCaches)):
        pos = masterXCaches[i]
        start = (pos-1) * 500
        end = pos*500
        partY = masterY[start:end]
        newY.extend(partY)
    print len(newY)
    newY = np.array(newY)
    print newY.shape
    print 'Loaded data'

    randomState = 51
    testSize = 0.1
    print testSize
    trainX, testX, trainY, testY = train_test_split(masterX, newY, test_size=testSize, random_state=randomState)
    trainY = trainY.reshape((trainY.shape[0], 1))
    testY = testY.reshape((testY.shape[0], 1))
    print trainX.shape
    print trainY.shape
    print testX.shape
    print testY.shape

    myNet = network(rows, cols)
    # Training
    model = tflearn.DNN(myNet, checkpoint_path='./model_resnet',
                        max_checkpoints=10, tensorboard_verbose=3, tensorboard_dir='./tflearn_logs')
    model.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY),
              show_metric=True, batch_size=32, run_id='resnet')
    model.save('./model_resnet/model1')

def evaluate(cacheN):
    rows = 224
    cols = 224
    numCaches = 32
    caches = range(numCaches)
    caches = [c+1 for c in caches]
    random.shuffle(caches)
    masterXCaches = [cacheN]
    masterX = loadDataFromCaches(masterXCaches, rows, cols)
    masterY = load_gt()
    newY = []
    for i in range(len(masterXCaches)):
        pos = masterXCaches[i]
        start = (pos-1) * 500
        end = pos*500
        partY = masterY[start:end]
        newY.extend(partY)
    print len(newY)
    newY = np.array(newY)
    print newY.shape
    print 'Loaded data'

    newY = newY.reshape((newY.shape[0], 1))
    
    myNet = network(rows, cols)
    model = tflearn.DNN(myNet)
    model.load('./model_resnet/model1')
    #print model.evaluate(masterX, newY, 16)

    predictedY = []
    for i in range(masterX.shape[0]/10):
        predictY = model.predict(masterX[i*10 : (i+1)*10])
        predictedY.extend(predictY)
    predictedY = np.array(predictedY)
    predictedY = predictedY.reshape((predictedY.shape[0], 1))
    #diffY = predictedY - newY
    denomY = []
    for i in range(newY.shape[0]):
        if newY[i] == 0:
            denomY.extend([1])
        else:
            denomY.extend(newY[i])
    denomY = np.array(denomY)
    denomY = denomY.reshape((newY.shape[0], 1))
    diffY = (predictedY - newY) / denomY
    diffYSqr = diffY ** 2

    if not os.path.isdir('./predictions'):
        os.mkdir('./predictions')
    predictionFile = open('./predictions/' + str(cacheN) + '.csv', 'w+')
    for i in range(predictedY.shape[0]):
        predictionFile.write(str(predictedY[i]) + ',' + str(newY[i]) + ',' + str(diffY[i]) + '\n')
    predictionFile.close()

    elemSum = np.sum(diffYSqr, axis=0)
    average = elemSum / diffY.shape[0]
    rmsd = np.sqrt(average)
    print rmsd


def display(cacheN):
    masterX = loadDataFromCaches([cacheN], 224, 224)
    masterY = load_gt()
    newY = []
    masterXCaches = [cacheN]
    for i in range(len(masterXCaches)):
        pos = masterXCaches[i]
        start = (pos-1) * 500
        end = pos*500
        partY = masterY[start:end]
        newY.extend(partY)
    print len(newY)
    newY = np.array(newY)
    print newY.shape
    print 'Loaded data'

    for i in range(masterX.shape[0]):
        img = np.array(masterX[i], dtype='uint8')
        x2 = 112
        y2 = 224
        m = float(newY[i])
        print m
        if m > 1.57:
            m = 1.57
        elif m < -1.57:
            m = -1.57
        print m
        angle = 0
        angle_debug = 0
        if m >= 0:
            angle_debug = 1.57 - m
            m = math.tan(angle_debug)
            print m
        else:
            angle_debug = 1.57 + m
            m = math.tan(angle_debug)

        angle = (1.57 - float(newY[i])) * 180.0 / 3.14
        x1 = (100.0 / math.sqrt(1 + m**2))
        y1 = m * float(x1)
        print('({}, {}) ({}, {}), m={}, angle={}, {}, rad={}'.format(x2, y2, x1, y1, m, angle, angle_debug, (1.57 - float(newY[i]))))
        y1 = 224 - y1
        if angle < 90:
            x1 = -1.0 * x1
        x1 = 112.0 * (1.0 + x1/224.0)
        print('({}, {}) ({}, {}), m={}, angle={}'.format(x2, y2, x1, y1, m, angle))
        print('\n')

        if angle < 30 or angle > 150:
            cv2.arrowedLine(img, (int(x2), int(y2)), (int(x1), int(y1)), (0, 0, 255), 5)
        else:
            cv2.arrowedLine(img, (int(x2), int(y2)), (int(x1), int(y1)), (255, 0, 0), 2)

        img = cv2.resize(img, (1280, 720))
        cv2.imshow('Video', img)
        cv2.waitKey(30)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    train_evaluate = sys.argv[1]
    if train_evaluate == '0':    
        train()
    elif train_evaluate == '1':
        cacheN = int(sys.argv[2])
        evaluate(cacheN)
    elif train_evaluate == '2':
        cacheN = int(sys.argv[2])
        display(cacheN)
