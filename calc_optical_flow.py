import numpy as np
import cv2
from train import loadDataFromCaches
import sys
from tqdm import tqdm

def calc_opticalflow(cacheNum):
    X = loadDataFromCaches(cacheNum, 224, 224)
    print X.shape

# Params for ShiTomasi corner detection
    featureParams = dict( maxCorners = 100, \
                          qualityLevel = 0.3, \
                          minDistance = 7, \
                          blockSize = 7 )

# Params for LK optical flow
    lkParams = dict( winSize = (15, 15), \
                     maxLevel = 2, \
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    frameNum = 0
# Create random colors
    colors = np.random.randint(0, 255, (100, 3))

# Find corners in the first image
    oldFrame = X[frameNum]
    frameNum += 1
    oldGray = cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(oldGray, mask = None, **featureParams)

# Mask image for drawing purpose
    mask = np.zeros_like(oldFrame)

    pb = tqdm(total=500)
    while(frameNum < 500):
        cv2.imshow('frame', X[frameNum])
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        frameNum += 1
        pb.update()

    pb.close()
    cv2.destroyAllWindows()

'''
    while(frameNum<500):
        frame = X[frameNum]
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Sake of understanding

        print oldFrame.shape
        print frame.shape
        print oldGray.shape
        print frameGray.shape
        cv2.imshow('old', oldGray)
        k = cv2.waitKey(0) & 0xff
        if k == 27:
            break
        cv2.imshow('new', frameGray)
        k = cv2.waitKey(0) & 0xff
        if k == 27:
            break
        oldGray = oldGray.reshape((224, 224, 1))
        frameGray = frameGray.reshape((224, 224, 1))
        oldGray = np.zeros((224, 224, 1))
        frameGray = np.zeros((224, 224, 1))

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(oldGray, frameGray, p0, None, **lkParams)

        # Understanding purpose
        print p1.shape
        print st.shape

        # Select good points
        goodNew = p1[st==1]
        goodOld = p0[st==1]

        # Draw the tracks
        for i, (new, old) in enumerate(zip(goodNew, goodOld)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a,b), (c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a,b), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)

        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Lets update the previous frame and previous points
        oldGray = framGray.copy()
        p0 = goodNew.reshape(-1, 1, 2)
        frameNum += 1

    cv2.destroyAllWindows()
'''


if __name__ == '__main__':
    cacheNum = sys.argv[1]
    calc_opticalflow(cacheNum) 

