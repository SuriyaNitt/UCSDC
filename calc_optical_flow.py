import numpy as np
import cv2
from train import loadDataFromCaches
import sys
from tqdm import tqdm


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def calc_opticalflow(cacheNum, typeOfFlow):
    X = loadDataFromCaches([cacheNum], 224, 224)
    X = np.array(X, dtype='uint8')
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

    '''
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
    if typeOfFlow == '1':
        while(frameNum<500):
            frame = X[frameNum]
            frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Sake of understanding

            print oldFrame.shape
            print frame.shape
            print oldGray.shape
            print frameGray.shape
            cv2.imshow('old', oldGray)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            cv2.imshow('new', frameGray)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            oldGray = oldGray.reshape((224, 224, 1))
            frameGray = frameGray.reshape((224, 224, 1))
            #oldGray = np.zeros((224, 224, 1))
            #frameGray = np.zeros((224, 224, 1))

            print frameGray.dtype
            print oldGray.dtype
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
                mask = cv2.line(mask, (a,b), (c,d), colors[i].tolist(), 2)
                frame = cv2.circle(frame, (a,b), 5, colors[i].tolist(), -1)
            img = cv2.add(frame, mask)

            cv2.imshow('frame', img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

            # Lets update the previous frame and previous points
            oldGray = frameGray.copy()
            p0 = goodNew.reshape(-1, 1, 2)
            frameNum += 1

        cv2.destroyAllWindows()

    elif typeOfFlow == '2':
        hsv = np.zeros((224, 224, 3), dtype='uint8')
        hsv[...,1] = 255
        print hsv.shape
        pb = tqdm(total=500)

        filenameSuffix = str(cacheNum) + '_' + str('0001')
        filename = './opticalflow/opticalflow_' + filenameSuffix + '.png'
        cv2.imwrite(filename, np.zeros((224, 224, 3)))

        while(frameNum<500):
            frame = X[frameNum]
            #fileName = "%04d" % (frameNum+1)
            #cv2.imwrite('./opticalflow_src/' + fileName + '.png', oldGray)
            frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imshow('old', oldGray)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            cv2.imshow('new', frameGray)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

            flow = cv2.calcOpticalFlowFarneback(oldGray, frameGray, \
                                                flow=None, \
                                                pyr_scale=0.5, \
                                                levels=3, \
                                                winsize=5, \
                                                iterations=3, \
                                                poly_n=3, \
                                                poly_sigma=1.2, \
                                                flags=0)

            #mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            #hsv[...,0] = ang*180/np.pi/2
            #hsv[...,2] = cv2.normalize(mag, None, 0, 2, cv2.NORM_MINMAX)
            #bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            bgr = draw_hsv(flow)
            drawFlow = draw_flow(frameGray, flow)

            formattedFrameNum = "%04d" % (frameNum+1)
            filenameSuffix = str(cacheNum) + '_' + formattedFrameNum
            filename = './opticalflow/opticalflow_' + filenameSuffix + '.png'
            cv2.imwrite(filename, drawFlow)

            #cv2.imshow('Difference', frameGray - oldGray)
            #cv2.imshow('New flow drawing', draw_hsv(flow))
            cv2.imshow('New flow drawing', drawFlow)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

            cv2.imshow('Dense optical flow', bgr)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            elif k == ord('s'):
                if not os.path.isdir('./optical_flow'):
                    os.path.mkdir('./optical_flow')
                filenameSuffix = str(cacheNum) + '_' + str(frameNum+1) + '_'
                stitched = np.append(frameGray, bgr, axis=0)
                cv2.imwrite('./opticalflow/stitched_' + filenameSuffix + '.png', stitched)
                #cv2.imwrite('opticalfb.png',frame2)
                #cv2.imwrite('opticalhsv.png',bgr)

            oldGray = frameGray.copy()
            pb.update()
            frameNum += 1
        pb.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    cacheNum = sys.argv[1]
    typeOfFlow = sys.argv[2] # 1 - sparse, 2 - dense
    calc_opticalflow(cacheNum, typeOfFlow)

