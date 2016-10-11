import os
import re


def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

myPath = './center_images'
imgFiles = os.listdir(myPath)
numImgFiles = len(imgFiles)
imgTimeStamp = []
for i in range(numImgFiles):
    timeStamp = imgFiles[i][7:]
    timeStamp = timeStamp[:-4]
    imgTimeStamp.append(timeStamp)

sort_nicely(imgTimeStamp)
imgTimeStamp = [float(myTime) for myTime in imgTimeStamp]
print imgTimeStamp
print len(imgTimeStamp)

imgHead = 0
steeringHead = 0
totalImgs = len(imgTimeStamp)
diff = 0
prevDiff = 10
prevTimeStamp = 0.0
prevSteeringAngle = 0.0

steeringInfo = open('./steeringinfo.csv')
groundTruth = open('./groundTruth.csv', 'w+')

line = steeringInfo.readline()
timeStamp = float(line.split(",")[0])
steeringAngle = float(line.split(",")[1])
print('{}, {}'.format(timeStamp, imgTimeStamp[imgHead]))
diff = abs(timeStamp - imgTimeStamp[imgHead])
prevDiff = diff
while line != '':
    line = steeringInfo.readline()
    timeStamp = float(line.split(",")[0])
    steeringAngle = float(line.split(",")[1])
    print('{}, {}'.format(timeStamp, imgTimeStamp[imgHead]))
    diff = abs(timeStamp - imgTimeStamp[imgHead])
    if diff > prevDiff:
        groundTruth.write(str(prevTimeStamp) + ',' + str(prevSteeringAngle) + '\n')
        imgHead += 1
        print('{}, {}'.format(timeStamp, imgTimeStamp[imgHead]))
        prevDiff = abs(timeStamp - imgTimeStamp[imgHead])
        print steeringHead
    else:
        prevTimeStamp = timeStamp
        prevSteeringAngle = steeringAngle
        prevDiff = diff
    steeringHead += 1

print steeringHead
steeringInfo.close()
groundTruth.close()