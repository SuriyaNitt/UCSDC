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
import glob

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
    progressBar = tqdm(total=70659)
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
    cache_data(imgs, './cache/' + str(142) + '.dat')
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

def load_prediction(cacheN):
    try:
        predictionFile = open('./predictions/' + str(cacheN) + '.csv')
    except:
        if cacheN == 31:
            predictions = np.zeros((282, 1))
        else:
            predictions = np.zeros((500, 1))
        return predictions

    line = predictionFile.readline()
    predictions = []
    while line != '':
        arr = line.split(',')
        predictions.append(float(arr[0]))
        line = predictionFile.readline()
    return predictions

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

def loadOpticalFlowImgs(masterXCaches, rows, cols):
    progressBar = tqdm(total=len(masterXCaches))
    opticalFlowPath = './opticalflow/flow/'
    imgs = np.ndarray((0, rows, cols, 3), dtype='float32')
    for i in range(len(masterXCaches)):
        imgFiles = glob.glob(opticalFlowPath + 'opticalflow_' + str(masterXCaches[i]) + '_*')
        imgFiles.sort(key=natural_keys)
        progressBar2 = tqdm(total=len(imgFiles))
        for imgFile in imgFiles:
            img = cv2.imread(imgFile)
            imgs = np.append(imgs, [img], axis=0)
            progressBar2.update()
        progressBar2.close()
        progressBar.update()
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

def double_network(rows, cols):
    # Building Residual Network
    net1 = tflearn.input_data(shape=[None, rows, cols, 3])
    net1 = tflearn.conv_2d(net1, 32, 3, activation='relu', bias=False)
    net1 = tflearn.max_pool_2d(net1, 2)
    # Residual blocks
    net1 = tflearn.residual_bottleneck(net1, 2, 16, 64, downsample=True)
    net1 = tflearn.residual_bottleneck(net1, 2, 32, 128, downsample=True)
    
    net1 = tflearn.batch_normalization(net1)
    net1 = tflearn.activation(net1, 'prelu')
    net1 = tflearn.global_avg_pool(net1)


    # Building Residual Network
    net2 = tflearn.input_data(shape=[None, rows, cols, 3])
    net2 = tflearn.conv_2d(net2, 32, 3, activation='relu', bias=False)
    net2 = tflearn.max_pool_2d(net2, 2)
    # Residual blocks
    net2 = tflearn.residual_bottleneck(net2, 2, 16, 64, downsample=True)
    net2 = tflearn.residual_bottleneck(net2, 2, 32, 128, downsample=True)
    
    net2 = tflearn.batch_normalization(net2)
    net2 = tflearn.activation(net2, 'prelu')
    net2 = tflearn.global_avg_pool(net2)

    # Merge layers
    net = tflearn.merge([net1, net2], mode='concat')
    
    # Add LSTM layers
    net = tflearn.reshape(net, new_shape=[1, 32, 256])
    net = tflearn.lstm(net, 128, return_seq=True)
    net = tflearn.lstm(net, 128)

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
    trainedData = []
    
    # this is for dataset 1
    #numCaches = 31
    #trainedData = [26, 4, 9, 29, 31]#[10, 15, 6, 27, 24]
    #trainedData.extend([10, 1, 3, 8, 28])
    # this is for dataset 2
    numCaches = 142
    trainedData = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 44, 49]

    caches = range(numCaches)
    caches = [c+1 for c in caches]
    random.shuffle(caches)
    for i in trainedData:
        caches.remove(i)
    masterXCaches = caches[0:1]
    masterXCaches = [53]
    print caches[0:1]
    masterX = loadDataFromCaches(masterXCaches, rows, cols)
    masterOFX = loadOpticalFlowImgs(masterXCaches, rows, cols)
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

    print masterOFX.shape

    randomState = 51
    testSize = 0.1
    print testSize

    '''
    referenceX = range(len(masterXCaches)*500)
    refTrainX, refTestX, trainY, testY = train_test_split(referenceX, newY, test_size=testSize, random_state=randomState)
    trainX = np.ndarray((0, 224, 224, 3), dtype='float32')
    optTrainX = np.ndarray((0, 224, 224, 3), dtype='float32')
    testX = np.ndarray((0, 224, 224, 3), dtype='float32')
    optTestX = np.ndarray((0, 224, 224, 3), dtype='float32')

    for i in refTrainX:
        trainX = np.append(trainX, [masterX[i]], axis=0)
        optTrainX = np.append(optTrainX, [masterOFX[i]], axis=0)
        #trainY = np.append(trainY, newY[i], axis=0)
    for i in refTestX:
        testX = np.append(testX, [masterX[i]], axis=0)
        optTestX = np.append(optTestX, [masterOFX[i]], axis=0)
        #testY = np.append(testY, newY[i], axis=0)
    

    trainX = masterX[:448]
    optTrainX = masterOFX[:448]
    testX = masterX[-64:]
    optTestX = masterOFX[-64:]

    trainY = newY[:448]
    testY = newY[-64:]
    '''
 
    trainX = np.ndarray((0, 224, 224, 3), dtype='float32')
    optTrainX = np.ndarray((0, 224, 224, 3), dtype='float32')
    testX = np.ndarray((0, 224, 224, 3), dtype='float32')
    optTestX = np.ndarray((0, 224, 224, 3), dtype='float32')

    trainY = []
    testY = []
    offset = 64

    #trainX = np.append(trainX, np.zeros)
    for i in range(32):
        trainX = np.append(trainX, masterX[i+offset : i+offset+32], axis=0)
        optTrainX = np.append(optTrainX, masterOFX[i+offset : i+offset+32], axis=0)
        for j in range(32):
            trainY.extend([newY[i+offset+31]])#newY[31:63]
    
    for i in range(16):
        testX = np.append(testX, masterX[416+i : 416+i+32], axis=0)
        optTestX = np.append(optTestX, masterOFX[416+i : 416+i+32], axis=0)
        for j in range(32):
            testY.extend([newY[i+448]])#newY[448:464]


    trainY = np.array(trainY)
    testY = np.array(testY)

    trainY = trainY.reshape((trainY.shape[0], 1))
    testY = testY.reshape((testY.shape[0], 1))
    print trainX.shape
    print trainY.shape
    print testX.shape
    print testY.shape

    myNet = double_network(rows, cols)
    # Training
    model = tflearn.DNN(myNet, checkpoint_path='./model_resnet',
                        max_checkpoints=10, tensorboard_verbose=3, tensorboard_dir='./tflearn_logs')
    #model.load('./model_resnet/model1')
    model.fit([trainX, optTrainX], trainY, n_epoch=6, validation_set=([testX, optTestX], testY), show_metric=True, batch_size=32, run_id='resnet')
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
    masterOFX = loadOpticalFlowImgs(masterXCaches, rows, cols)
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
    
    myNet = double_network(rows, cols)
    model = tflearn.DNN(myNet)
    model.load('./model_resnet/model1')
    #print model.evaluate(masterX, newY, 16)

    predictedY = []
    progressBar = tqdm(total=468)
    for i in range(masterX.shape[0] - 31):
        #predictY = model.predict([masterX[i*32 : (i+1)*32], masterOFX[i*32 : (i+1)*32]])
        predictY = model.predict([masterX[i : i+32], masterOFX[i : i+32]])
        predictedY.extend(predictY[0])
        progressBar.update()
    for i in range(31):
        predictedY.extend([0.0])
    progressBar.close()
    predictedY = np.array(predictedY)
    print predictedY.shape
    print predictedY
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
    print predictedY.shape
    print newY.shape
    diffY = np.subtract(predictedY, newY)
    diffYSqr = np.square(diffY)

    if not os.path.isdir('./predictions'):
        os.mkdir('./predictions')
    predictionFile = open('./predictions/' + str(cacheN) + '.csv', 'w+')
    for i in range(predictedY.shape[0]):
        predictionFile.write(str(predictedY[i][0]) + ',' + str(newY[i][0]) + ',' + str(diffY[i][0]) + '\n')
    predictionFile.close()

    elemSum = np.sum(diffYSqr, axis=0)
    average = elemSum / diffY.shape[0]
    rmsd = np.sqrt(average)
    rmsd_degrees = rmsd * 180.0 / np.pi
    print ('rmsd_radians:{}, rmsd_degrees:{}'.format(rmsd, rmsd_degrees))


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

    predictionY = load_prediction(cacheN)
    predictionY = np.array(predictionY)

    for i in range(masterX.shape[0]):
        img = np.array(masterX[i], dtype='uint8')
        x2 = 112
        y2 = 224
        m = float(newY[i])
        mPredicted = float(predictionY[i])

        angle = 0
        angle_debug = 0

        #Part1
        if m > 1.57:
            m = 1.57
        elif m < -1.57:
            m = -1.57

        #Part2
        if m >= 0:
            angle_debug = 1.57 - m
            m = math.tan(angle_debug)
        else:
            angle_debug = 1.57 + m
            m = math.tan(angle_debug)

        #Part3
        angle = (1.57 - float(newY[i])) * 180.0 / 3.14
        x1 = (100.0 / math.sqrt(1 + m**2))
        y1 = 224 - m * float(x1)
        if angle < 90:
            x1 = -1.0 * x1
        x1 = 112.0 * (1.0 + x1/224.0)

        #Part4
        if angle < 30 or angle > 150:
            cv2.arrowedLine(img, (int(x2), int(y2)), (int(x1), int(y1)), (0, 0, 255), 5)
        else:
            cv2.arrowedLine(img, (int(x2), int(y2)), (int(x1), int(y1)), (255, 0, 0), 2)

        cv2.putText(img, 'GroundTruth', (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 0), 1)

        #----------------------------------------------------------------------------------------

        anglePredicted = 0
        angle_debug_predicted = 0

        #Prediction part1
        if mPredicted > 1.57:
            mPredicted = 1.57
        elif mPredicted < -1.57:
            mPredicted = -1.57
        #Prediction part2
        if mPredicted >= 0:
            angle_debug_predicted = 1.57 - mPredicted
            mPredicted = math.tan(angle_debug_predicted)
        else:
            angle_debug = 1.57 + mPredicted
            mPredicted = math.tan(angle_debug_predicted)

        #Prediction part3
        anglePredicted = (1.57 - float(predictionY[i])) * 180.0 / 3.14
        x3 = (100.0 / math.sqrt(1 + mPredicted**2))
        y3 = 224 - mPredicted * float(x3)
        if anglePredicted < 90:
            x3 = -1.0 * x3
        x3 = 112.0 * (1.0 + x3/224.0)

        #Prediction part4
        if anglePredicted < 30 or anglePredicted > 150:
            cv2.arrowedLine(img, (int(x2), int(y2)), (int(x3), int(y3-50)), (0, 255, 255), 5)
        else:
            cv2.arrowedLine(img, (int(x2), int(y2)), (int(x3), int(y3-50)), (0, 255, 0), 2)

        cv2.putText(img, 'Prediction', (int(x3), int(y3-50)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 0), 1)

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
