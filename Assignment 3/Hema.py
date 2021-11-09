import os
from sklearn.svm import LinearSVC, SVC
from skimage.feature import hog
from sklearn.model_selection import train_test_split
import cv2
from sklearn.multiclass import OneVsRestClassifier

TRAIN_PATH = 'D:\ML Assignments\Assignment 3\train'
train_images = [TRAIN_PATH + i for i in os.listdir(TRAIN_PATH)]

TEST_PATH = 'D:\ML Assignments\Assignment 3\try1'
test_images = [TEST_PATH + i for i in os.listdir(TEST_PATH)]

X = []
y = []

def getFeatures():
    for pic in os.listdir(TRAIN_PATH):
        pic_data = cv2.imread('train/' + pic, cv2.IMREAD_UNCHANGED)
        resizedPic = cv2.resize(pic_data, (128, 64))
        picFetures, hog_image = hog(resizedPic, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True,
                            multichannel=True)
        X.append(picFetures)
    return X


getFeatures()
def fillYLabels():
    for pic in os.listdir(TRAIN_PATH):
        if 'cat' in pic:
            y.append(0)
        else:
            y.append(1)
    return y

fillYLabels()
def printAccuracy(prediction, goal):
    rightPredictedCounter = 0
    newResults = []
    for i in range(len(goal)):
        if float(prediction[i]) > 0.5:
            newResults.append(1)
        else:
            newResults.append(0)

        if newResults[i] == goal[i]:
            rightPredictedCounter += 1
    print('Accuracy:')
    print(str(rightPredictedCounter) + '/' + str(len(goal)) + '*100=')
    print(str(float((rightPredictedCounter / len(goal) * 100))) + '%')

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

def linearSVCOVO():
    svc = LinearSVC()
    x = getFeatures()
    y = fillYLabels()
    svc.fit(x, y)
    pred = svc.predict(x_test)
    print('Model')
    printAccuracy(pred, y_test)

def linearSVCOVR():
    svc = OneVsRestClassifier(SVC(kernel='linear', C=0.4))
    x = getFeatures()
    y = fillYLabels()
    svc.fit(x, y)
    pred = svc.predict(x_test)
    print('Model')
    printAccuracy(pred, y_test)

def polynomialSVCOVO():
    svc = SVC(kernel='poly', C=0.6)
    x = getFeatures()
    y = fillYLabels()
    svc.fit(x, y)
    pred = svc.predict(x_test)
    print('Model')
    printAccuracy(pred, y_test)

def polynomialSVCOVR():
    svc = OneVsRestClassifier(SVC(kernel='linear', C=0.6))
    x = getFeatures()
    y = fillYLabels()
    svc.fit(x, y)
    pred = svc.predict(x_test)
    print('Model')
    printAccuracy(pred, y_test)


def RBFSVCOVO():
    svc = SVC(C=0.8)
    x = getFeatures()
    y = fillYLabels()
    svc.fit(x, y)
    pred = svc.predict(x_test)
    print('Model')
    printAccuracy(pred, y_test)

def RBFSVCOVR():
    svc = OneVsRestClassifier(SVC(C=0.8))
    x = getFeatures()
    y = fillYLabels()
    svc.fit(x, y)
    pred = svc.predict(x_test)
    print('Model')
    printAccuracy(pred, y_test)

linearSVCOVO()
linearSVCOVR()
polynomialSVCOVO()
polynomialSVCOVR()
RBFSVCOVO()
RBFSVCOVR()
