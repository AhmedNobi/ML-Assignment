import os
from sklearn.svm import LinearSVC, SVC
from skimage.feature import hog
from sklearn.model_selection import train_test_split
import cv2
from sklearn.multiclass import OneVsRestClassifier

train_test_loc = 'train'
train_images = [train_test_loc + i for i in os.listdir(train_test_loc)]

training_data = []
y = []

def getFeatures():
    for pic in os.listdir(train_test_loc):
        pic_data = cv2.imread('train/' + pic, cv2.IMREAD_UNCHANGED)
        resizedPic = cv2.resize(pic_data, (128, 64))
        picFetures, hog_image = hog(resizedPic, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True,
                            multichannel=True)
        training_data.append(picFetures)
    return training_data


getFeatures()
def fillYLabels():
    for pic in os.listdir(train_test_loc):
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

x_train, x_test, y_train, y_test = train_test_split(training_data, y, test_size=0.2, random_state=0, shuffle=True)


def choose(num):
    if num == 1:
        linearSVCOVO()
    elif num == 2:
        kernel_linear_ovr()
    elif num == 3:
        kernel_poly_ovo()
    elif num == 4:
        kernel_poly_ovr()
    elif num == 5:
        kernel_gauss_ovo()
    elif num == 6:
        kernel_gauss_ovr()


def linearSVCOVO():
    svc = LinearSVC()
    x = getFeatures()
    y = fillYLabels()
    svc.fit(x, y)
    pred = svc.predict(x_test)
    print('Model')
    printAccuracy(pred, y_test)

def kernel_linear_ovr():
    svc = OneVsRestClassifier(SVC(kernel='linear', C=0.4))
    x = getFeatures()
    y = fillYLabels()
    svc.fit(x, y)
    pred = svc.predict(x_test)
    print('Model')
    printAccuracy(pred, y_test)

def kernel_poly_ovo():
    svc = SVC(kernel='poly', C=0.6)
    x = getFeatures()
    y = fillYLabels()
    svc.fit(x, y)
    pred = svc.predict(x_test)
    print('Model')
    printAccuracy(pred, y_test)

def kernel_poly_ovr():
    svc = OneVsRestClassifier(SVC(kernel='linear', C=0.6))
    x = getFeatures()
    y = fillYLabels()
    svc.fit(x, y)
    pred = svc.predict(x_test)
    print('Model')
    printAccuracy(pred, y_test)


def kernel_gauss_ovo():
    svc = SVC(C=0.8)
    x = getFeatures()
    y = fillYLabels()
    svc.fit(x, y)
    pred = svc.predict(x_test)
    print('Model')
    printAccuracy(pred, y_test)

def kernel_gauss_ovr():
    svc = OneVsRestClassifier(SVC(C=0.8))
    x = getFeatures()
    y = fillYLabels()
    svc.fit(x, y)
    pred = svc.predict(x_test)
    print('Model')
    printAccuracy(pred, y_test)

choose(1)
