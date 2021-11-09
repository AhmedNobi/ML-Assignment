import os
from sklearn.svm import LinearSVC, SVC
from skimage.feature import hog
from sklearn.model_selection import train_test_split
import cv2
from sklearn.multiclass import OneVsRestClassifier

TRAIN_PATH = 'try'
train_images = [TRAIN_PATH + i for i in os.listdir(TRAIN_PATH)]

features = []
target = []

def features_extraction():
    for image in os.listdir(TRAIN_PATH):
        img_data = cv2.imread('train/' + image, cv2.IMREAD_UNCHANGED)
        resize = cv2.resize(img_data, (128, 64))
        feutres_from_image, hog_image = hog(resize, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True,
                            multichannel=True)
        features.append(feutres_from_image)
    return features

def train():
    for i in os.listdir(TRAIN_PATH):
        if 'dog' in i:
            target.append(1)
        else:
            target.append(0)
    return target

def get_results(y_predict, y_actual):
    total = 0
    prediction_results = []
    for i in range(len(y_actual)):
        if float(y_predict[i]) >= 0.5:
            prediction_results.append(1)
        else:
            prediction_results.append(0)

        if prediction_results[i] == y_actual[i]:
            total += 1
    print('Accuracy: ' + str(total) + ' / ' + str(len(y_actual)) + ' * 100 = ' + str(float((total / len(y_actual) * 100))) + '%')

def set_data():
    features_extraction()
    train()
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=0, shuffle=True)
    return x_train, x_test, y_train, y_test

def apply_linear_svc(one_vs_one=True):
    print('One Vs One is ' + str(one_vs_one))
    if one_vs_one:
        svc = LinearSVC()
    else:
        svc = OneVsRestClassifier(SVC(kernel='linear', C=1))
    svc.fit(features_extraction(), train())
    pred = svc.predict(x_test)
    print('Linear SVC')
    get_results(pred, y_test)

def apply_polynomial_svc(one_vs_one=True):
    print('One Vs One is ' + str(one_vs_one))
    if one_vs_one:
        svc = SVC(kernel='poly', C=1)
    else:
        svc = OneVsRestClassifier(SVC(kernel='linear', C=1))
    svc.fit(features_extraction(), train())
    pred = svc.predict(x_test)
    print('Polynomial SVC')
    get_results(pred, y_test)

def apply_gaussian_svc(one_vs_one=True):
    print('One Vs One is ' + str(one_vs_one))
    if one_vs_one:
        svc = SVC(C=1)
    else:
        svc = OneVsRestClassifier(SVC(C=1))
    svc.fit(features_extraction(), train())
    pred = svc.predict(x_test)
    print('RBF SVC')
    get_results(pred, y_test)

x_train, x_test, y_train, y_test = set_data()

apply_linear_svc(one_vs_one=False)
apply_linear_svc()

apply_polynomial_svc(one_vs_one=False)
apply_polynomial_svc()

apply_gaussian_svc(one_vs_one=False)
apply_gaussian_svc()
