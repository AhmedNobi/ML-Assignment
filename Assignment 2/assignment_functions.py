import numpy as np
## Support Vector Machine
def fit(X,Y):
    X = np.c_[np.ones((X.shape[0], 1)), X]
    w = np.zeros(X[0].shape)
    epochs = 1
    alpha = 0.001
    while epochs < 10000:
        lambda_ = 1 / epochs
        for i in range(100):
            y_predict_temp = np.dot(w.T, X[i])
            for j in range(3):
                if Y[i] * y_predict_temp >= 1:
                    w[j] = w[j] - alpha * 2 * lambda_ * w[j]
                else:
                    w[j] = w[j] + alpha * (Y[i] * X[i, j] - 2 * lambda_ * w[j])
        epochs += 1
    y_pred = []
    for i in range(100):
        y_predict_temp = np.dot(w.T, X[i])
        if y_predict_temp >= 0:
            y_pred.append(1)
        else:
            y_pred.append(-1)
    return y_pred, w
