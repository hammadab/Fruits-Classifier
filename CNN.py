import time

blah = time.time()
import cv2
import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as pyplot
from sklearn.metrics import confusion_matrix

data_set = 1

# def conv2(image, filtr, bias):
#     (nf, fw, fh) = filtr.shape
#     (ni, iw, ih, ic) = image.shape
#     y = np.zeros((ni, nf, (iw - fw + 1), (ih - fh + 1), ic))
#     for n in range(0, ni): # no. of samples
#         for i in range(0, nf): # no. of filter
#             for j in range(0, (iw - fw + 1)):
#                 for k in range(0, (ih - fh + 1)):
#                     for l in range(0, ic): # RGB channel
#                         y[n, i, j, k, l] = np.sum(image[n, j : j + fw, k : k + fh, l] * filtr[i]) + bias[i]
#     return y

def conv2_train(image, filtr, bias, error, lr):
    (nf, fw, fh) = filtr.shape
    (ni, iw, ih, ic) = image.shape
    error = error.reshape((ni, nf, (iw - fw + 1), (ih - fh + 1), ic))
    df = np.zeros(filtr.shape)
    # db = np.sum(np.sum(np.sum(np.sum(error, axis=0), axis=1), axis=1), axis=1)  # gradient of bias
    db = np.sum(error, axis=(0, 2, 3, 4))  # gradient of bias
    for i in range(0, nf):  # no. of filter
        for j in range(0, iw - fw + 1):
            for k in range(0, ih - fh + 1):
                for l in range(0, ic):  # RGB channel
                    df[i] += np.matmul(error[:, i, j, k, l].T,
                                       image[:, j: j + fw, k: k + fh, l].reshape((ni, -1))).reshape(
                        (fw, fh))  # gradient of filter
    filtr -= lr * df

    bias -= lr * db.reshape((-1, 1))
    ## FOR DEBUGGING #### FOR DEBUGGING #### FOR DEBUGGING #### FOR DEBUGGING #### FOR DEBUGGING #### FOR DEBUGGING #### FOR DEBUGGING #### FOR DEBUGGING #### FOR DEBUGGING ##
    # fs = np.sum(np.abs(df))
    # fbs = np.sum(np.abs(db))
    # if fs < 1 or fbs < 1:
    #     print('f', fs)
    #     print('fb', fbs)
    #     # rukjao
    ## FOR DEBUGGING #### FOR DEBUGGING #### FOR DEBUGGING #### FOR DEBUGGING #### FOR DEBUGGING #### FOR DEBUGGING #### FOR DEBUGGING #### FOR DEBUGGING #### FOR DEBUGGING ##
    return filtr, bias


if data_set == 1:
    grayscale = True
    trainImages = []
    trainLabels = []
    valImages = []
    valLabels = []
    image_dimensions = (50, 50)  # should be less than original (100, 100)
    i = 0  # label
    for folder in [x[0] for x in os.walk('./fruits-360/Training')][1:]:
        j = 0
        for filename in os.listdir(folder):
            j += 1
            temp = cv2.imread(os.path.join(folder, filename))
            temp = cv2.resize(temp, image_dimensions)
            if grayscale:
                temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)  # convert to grayscale
                # temp = temp.reshape((temp.shape[0], temp.shape[1], 1))
            ## FOR DEBUGGING #### FOR DEBUGGING #### FOR DEBUGGING #### FOR DEBUGGING #### FOR DEBUGGING #### FOR DEBUGGING #### FOR DEBUGGING #### FOR DEBUGGING #### FOR DEBUGGING ##
            # if j <= 9:
            #     trainImages.append(temp)
            #     trainLabels.append(i)
            # elif j <= 10:
            #     valImages.append(temp)
            #     valLabels.append(i)
            # else:
            #     break
            ## FOR DEBUGGING #### FOR DEBUGGING #### FOR DEBUGGING #### FOR DEBUGGING #### FOR DEBUGGING #### FOR DEBUGGING #### FOR DEBUGGING #### FOR DEBUGGING #### FOR DEBUGGING ##
            if j <= 450:  # 90 %
                trainImages.append(temp)
                trainLabels.append(i)
            else:
                valImages.append(temp)
                valLabels.append(i)
        i += 1  # label
    num_of_classes = i
    trainImages = np.array(trainImages)
    trainLabels = np.array(trainLabels)
    valImages = np.array(valImages)
    valLabels = np.array(valLabels)
else:
    trainData = np.genfromtxt('./mnist_train.csv', delimiter=',')[1:]
    num_of_classes = 10
    trainImages = np.array(trainData[0: (9 * trainData.shape[0] // 10), 1:])
    trainLabels = np.array(trainData[0: (9 * trainData.shape[0] // 10), 0], dtype=int)
    valImages = np.array(trainData[0: (trainData.shape[0] // 10), 1:])
    valLabels = np.array(trainData[0: (trainData.shape[0] // 10), 0], dtype=int)
    trainImages = trainImages.reshape((trainImages.shape[0], 28, 28, 1))
    valImages = valImages.reshape((valImages.shape[0], 28, 28, 1))
    del trainData

permutation = np.random.permutation(len(trainLabels))
trainImages = trainImages[permutation]  # shuffle train images
trainLabels = trainLabels[permutation]  # shuffle train labels
del permutation

trainImages = trainImages / 255  # normalize # needs lots of RAM
valImages = valImages / 255  # normalize # needs lots of RAM

if trainImages.shape != 4 or valImages.shape != 4:
    try:
        trainImages = trainImages.reshape((trainImages.shape[0], trainImages.shape[1], trainImages.shape[2], 1))
        valImages = valImages.reshape((valImages.shape[0], valImages.shape[1], valImages.shape[2], 1))
        print("Assuming images are grayscale, they're reshaped into 4D (samples X height X width X channels)")
    except:
        raise NameError('Images need to be in 4D shape (samples X height X width X channels)')

trainLabels = np.eye(num_of_classes)[trainLabels]  # one-hot encoding
valLabels = np.eye(num_of_classes)[valLabels]  # one-hot encoding

# initialize
# sd = 0.01
filter_dim = (5, 10, 10)  # (channels X filter height X filter width)
if filter_dim[1] > trainImages.shape[1] or filter_dim[2] > trainImages.shape[2]:
    raise NameError("Filters are required to be smaller than the image")

filtr = np.random.rand(filter_dim[0], filter_dim[1], filter_dim[2]) / (
            filter_dim[1] * filter_dim[2])  # 5 filters of 10x10
bias = np.random.rand(filter_dim[0], 1) / (filter_dim[1] * filter_dim[2])  # bias for filters
h_size = filtr.shape[0] * (trainImages.shape[1] - filtr.shape[1] + 1) * (trainImages.shape[2] - filtr.shape[2] + 1) * \
         trainImages.shape[3]
b = np.random.rand(num_of_classes, 1) / h_size  # .astype('float64')  # initialize bias for output layer
w = np.random.rand(num_of_classes, h_size) / h_size  # .astype('float64')  # initialize weights for output layer

epochs = 5
m = 512  # batch size
lr = 0.01  # learning rate

TrMSE = np.zeros((epochs, 1))  # Training Mean Squared Error
VaMSE = np.zeros((epochs, 1))  # Validation Mean Squared Error

TrCA = np.zeros((epochs, 1))  # Training Mean Classification Accuracy
VaCA = np.zeros((epochs, 1))  # Validation Classification Accuracy


#### ETA #### ETA #### ETA #### ETA #### ETA #### ETA #### ETA #### ETA #### ETA #### ETA #### ETA #### ETA #### ETA #### ETA #### ETA #### ETA ####
# a = time.time()

# aa = ((time.time() - a) / m) * ((len(trainLabels) + len(valLabels)) * (epochs + 1) + 20618)
# if aa > 3600:
#     print("ETA", '%.2f' % (aa / 3600), "hour(s)")
# elif aa > 60:
#     print("ETA", '%.2f' % (aa / 60), "minutes")
# else:
#     print("ETA", '%.2f' % aa, "seconds")
#### ETA #### ETA #### ETA #### ETA #### ETA #### ETA #### ETA #### ETA #### ETA #### ETA #### ETA #### ETA #### ETA #### ETA #### ETA #### ETA ####
def asd(qwe):
    print('mean', np.mean(qwe))
    print('min', np.min(qwe))
    print('max', np.max(qwe))


for epoch in range(0, epochs):
    a = time.time()
    permutation = np.random.permutation(len(trainLabels))
    trainImages = trainImages[permutation]  # shuffle train images
    trainLabels = trainLabels[permutation]  # shuffle train labels
    del permutation
    for i in range(0, int(np.ceil(len(trainLabels) / m))):
        # print('w', np.sum(np.abs(w)))
        # asd(w)
        # print('b', np.sum(np.abs(b)))
        # asd(b)
        # print('filtr', np.sum(np.abs(filtr)))
        # asd(filtr)
        # print('bias', np.sum(np.abs(bias)))
        # asd(bias)
        # Forward pass
        x = trainImages[(i * m):(i + 1) * m]  # .astype('float64')
        # h = conv2(x, filtr, bias)
        h = []
        for j in range(0, filtr.shape[0]):
            temp = []
            for l in range(0, x.shape[3]):  # RGB channel
                ntemp = []
                for n in range(0, x.shape[0]):
                    ntemp.append(signal.convolve2d(x[n, :, :, l], filtr[j], mode='valid') + bias[j])
                temp.append(np.array(ntemp))
            h.append(np.array(temp))
        h = np.transpose(np.array(h), (2, 0, 3, 4, 1))  # .astype('float64')
        h = h.reshape((x.shape[0], -1))
        # print('h', np.sum(np.abs(h)))
        # asd(h)
        # h[h < 0] = 0  # ReLu
        h = 1 / (1 + np.exp(-h))  # , dtype='float64'))  # sigmoid
        # print('h relu', np.sum(np.abs(h)))
        # asd(h)
        whb = h.dot(w.T) + b.T
        # print('whb', np.sum(np.abs(whb)))
        # yc = whb
        # asd(whb)
        yc = np.exp(whb)  # , dtype='float64'))  # Softmax
        # print('yc', np.sum(np.abs(yc)))
        # asd(yc)
        yc /= np.sum(yc, axis=1).reshape((x.shape[0], 1))  # calculate output
        # yc = 1 / (1 + np.exp(-whb))  # sigmoid output
        # print('yc', np.sum(np.abs(yc)))
        # yc = np.nan_to_num(yc)
        # print('yc', np.sum(np.abs(yc)))
        yt = trainLabels[(i * m):((i + 1) * m)]  # known true output

        # Backpropagation
        error = (yc - yt) / x.shape[0]
        # delta = error * yc * (1 - yc)  # sigmoid derivative
        # gws = np.sum(np.abs(np.matmul(delta.T, h)))
        # gbs = np.sum(np.abs(delta))
        # if gws < 1 or gbs < 1:
        #     print('gw', gws)
        #     print('gb', gbs)
        #     # rukjao
        w -= lr * np.matmul(error.T, h)
        b -= lr * np.sum(error, axis=0).reshape(b.shape)
        # print('w', np.sum(np.abs(w)))
        # print('error', np.sum(np.abs(error)))
        conv_error = error.dot(w)
        conv_error[h < 0] = 0  # ReLu derivative
        # conv_error = conv_error * h * (1 - h)  # sigmoid derivative
        # print('conv_error', np.sum(np.abs(conv_error)))
        filtr, bias = conv2_train(x, filtr, bias, conv_error, lr)
        # rukjao
        # Performance check
        TrMSE[epoch] += np.sum((yt - yc) ** 2)  # Training Squared Error
        TrCA[epoch] += np.sum(np.argmax(yc, axis=1) == np.argmax(yt, axis=1))
    TrMSE[epoch] = TrMSE[epoch] / (2 * len(trainLabels))
    TrCA[epoch] = 100 * TrCA[epoch] / len(trainLabels)

    # Validation
    x = valImages  # .astype('float64')
    # h = conv2(x, filtr, bias)
    h = []
    for j in range(0, filtr.shape[0]):
        temp = []
        for l in range(0, x.shape[3]):  # RGB channel
            ntemp = []
            for n in range(0, x.shape[0]):
                ntemp.append(signal.convolve2d(x[n, :, :, l], filtr[j], mode='valid') + bias[j])
            temp.append(np.array(ntemp))
        h.append(np.array(temp))
    h = np.transpose(np.array(h), (2, 0, 3, 4, 1))
    h = h.reshape((x.shape[0], -1))
    h[h < 0] = 0  # ReLu
    # h = 1 / (1 + np.exp(-h))#, dtype='float64'))  # sigmoid
    yc = (np.exp(h.dot(w.T) + b.T))  # , dtype='float64'))  # Softmax
    yc /= np.sum(yc, axis=1).reshape((x.shape[0], 1))  # calculate output
    # yc = 1 / (1 + np.exp(-(h.dot(w.T) + b.T)))  # sigmoid output
    yt = valLabels  # known true output
    VaMSE[epoch] += np.sum((yt - yc) ** 2)  # Validation Squared Error
    VaCA[epoch] += np.sum(np.argmax(yc, axis=1) == np.argmax(yt, axis=1))
    VaMSE[epoch] = VaMSE[epoch] / (2 * len(valLabels))
    VaCA[epoch] = 100 * VaCA[epoch] / len(valLabels)

    # # Save model
    # with open('./' + str(epoch) + 'cnn.npy', 'wb') as f:
    #     np.save(f, filtr)
    #     np.save(f, bias)
    #     np.save(f, b)
    #     np.save(f, w)
    #     np.save(f, TrMSE)
    #     np.save(f, VaMSE)
    #     np.save(f, TrCA)
    #     np.save(f, VaCA)

    # Time
    aa = time.time() - a
    if aa > 3600:
        print("epoch", epoch + 1, "took", '%.2f' % (aa / 3600), "hour(s)")
    elif aa > 60:
        print("epoch", epoch + 1, "took", '%.2f' % (aa / 60), "minutes")
    else:
        print("epoch", epoch + 1, "took", '%.2f' % aa, "seconds")
    aa = (epochs - epoch) * aa
    if aa > 3600:
        print("ETA", '%.2f' % (aa / 3600), "hour(s)")
    elif aa > 60:
        print("ETA", '%.2f' % (aa / 60), "minutes")
    else:
        print("ETA", '%.2f' % aa, "seconds")

pyplot.figure()
pyplot.plot(TrMSE, label="Training")
pyplot.plot(VaMSE, label="Validation")
pyplot.title("Mean Squared Error")
pyplot.ylabel("error")
pyplot.xlabel('epoch')
pyplot.legend(loc='best')
pyplot.savefig("./CNN MSE.png")
pyplot.figure()
pyplot.plot(TrCA, label="Training")
pyplot.plot(VaCA, label="Validation")
pyplot.title("Classification Accuracy")
pyplot.ylabel("%")
pyplot.xlabel('epoch')
pyplot.legend(loc='best')
pyplot.savefig("./CNN CA.png")

pyplot.figure()
pyplot.imshow(confusion_matrix(np.argmax(yc, axis=1), np.argmax(yt, axis=1)))
pyplot.savefig("./CNN val CM.png")

x = trainImages.astype('float64')
# h = conv2(x, filtr, bias)
h = []
for j in range(0, filtr.shape[0]):
    temp = []
    for l in range(0, x.shape[3]):  # RGB channel
        ntemp = []
        for n in range(0, x.shape[0]):
            ntemp.append(signal.convolve2d(x[n, :, :, l], filtr[j], mode='valid') + bias[j])
        temp.append(np.array(ntemp))
    h.append(np.array(temp))
h = np.transpose(np.array(h), (2, 0, 3, 4, 1))
h = h.reshape((x.shape[0], -1))
h[h < 0] = 0  # ReLu
# h = 1 / (1 + np.exp(-h, dtype='float64'))  # sigmoid
yc = (np.exp(h.dot(w.T) + b.T, dtype='float64'))  # Softmax
yc /= np.sum(yc, axis=1).reshape((x.shape[0], 1))  # calculate output
# yc = 1 / (1 + np.exp(-(h.dot(w.T) + b.T)))  # sigmoid output
pyplot.figure()
pyplot.imshow(confusion_matrix(np.argmax(yc, axis=1), np.argmax(trainLabels, axis=1)))
pyplot.savefig("./CNN train CM.png")

if data_set == 1:
    testImages = []
    testLabels = []
    i = 0  # label
    for folder in [x[0] for x in os.walk('./fruits-360/Test')][1:]:
        for filename in os.listdir(folder):
            temp = cv2.imread(os.path.join(folder, filename))
            temp = cv2.resize(temp, image_dimensions)
            if grayscale:
                temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            testImages.append(temp)
            testLabels.append(i)
        i += 1  # label
    testImages = np.array(testImages)
    testLabels = np.array(testLabels)
else:
    testData = np.genfromtxt('./mnist_test.csv', delimiter=',')[1:]
    testImages = np.array(trainData[0: (9 * testData.shape[0] // 10), 1:])
    testLabels = np.array(trainData[0: (9 * testData.shape[0] // 10), 0], dtype=int)
    testImages = testImages.reshape((testImages.shape[0], 28, 28, 1))
    del testData

testImages = testImages / 255  # normalize # needs lots of RAM
testLabels = np.eye(num_of_classes)[testLabels]  # one-hot encoding

if testImages.shape != 4:
    try:
        testImages = testImages.reshape((testImages.shape[0], testImages.shape[1], testImages.shape[2], 1))
        print("Assuming images are grayscale, they're reshaped into 4D (samples X height X width X channels)")
    except:
        raise NameError('Images need to be in 4D shape (samples X height X width X channels)')

x = testImages.astype('float64')
# h = conv2(x, filtr, bias)
h = []
for j in range(0, filtr.shape[0]):
    temp = []
    for l in range(0, x.shape[3]):  # RGB channel
        ntemp = []
        for n in range(0, x.shape[0]):
            ntemp.append(signal.convolve2d(x[n, :, :, l], filtr[j], mode='valid') + bias[j])
        temp.append(np.array(ntemp))
    h.append(np.array(temp))
h = np.transpose(np.array(h), (2, 0, 3, 4, 1))
h = h.reshape((x.shape[0], -1))
h[h < 0] = 0  # ReLu
# h = 1 / (1 + np.exp(-h, dtype='float64'))  # sigmoid
yc = (np.exp(h.dot(w.T) + b.T, dtype='float64'))  # Softmax
yc /= np.sum(yc, axis=1).reshape((x.shape[0], 1))  # calculate output
# yc = 1 / (1 + np.exp(-(h.dot(w.T) + b.T,)))  # sigmoid output
pyplot.figure()
pyplot.imshow(confusion_matrix(np.argmax(yc, axis=1), np.argmax(testLabels, axis=1)))
pyplot.savefig("./CNN test CM.png")

aa = time.time() - blah
if aa > 3600:
    print("Time Elapsed to run whole code", '%.2f' % (aa / 3600), "hour(s)")
elif aa > 60:
    print("Time Elapsed to run whole code", '%.2f' % (aa / 60), "minutes")
else:
    print("Time Elapsed to run whole code", '%.2f' % aa, "seconds")
