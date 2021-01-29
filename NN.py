import time

blah = time.time()
import cv2
import os
import numpy as np
import matplotlib.pyplot as pyplot
from sklearn.metrics import confusion_matrix

data_set = 0

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
    del trainData

permutation = np.random.permutation(len(trainLabels))
trainImages = trainImages[permutation]  # shuffle train images
trainLabels = trainLabels[permutation]  # shuffle train labels
del permutation

trainImages = trainImages / 255  # normalize # needs lots of RAM
valImages = valImages / 255  # normalize # needs lots of RAM
trainImages = trainImages.reshape((trainImages.shape[0], -1))  # vectorize each observation
valImages = valImages.reshape((valImages.shape[0], -1))  # vectorize each observation

trainLabels = np.eye(num_of_classes)[trainLabels]  # one-hot encoding
valLabels = np.eye(num_of_classes)[valLabels]  # one-hot encoding

# sd = 1
# for n in [0.1, 0.001, 0.00001]:
#     for sd in [0.1, 0.01, 0.001]:
b1 = np.random.rand(960, 1) / trainImages.shape[1]  # initialize bias for hidden layer 1
w1 = np.random.rand(960, trainImages.shape[1]) / trainImages.shape[1]  # initialize weights for hidden layer 1
b2 = np.random.rand(480, 1) / 960  # initialize bias for hidden layer 2
w2 = np.random.rand(480, 960) / 960  # initialize weights for hidden layer 2
b3 = np.random.rand(240, 1) / 480  # initialize bias for hidden layer 3
w3 = np.random.rand(240, 480) / 480  # initialize weights for hidden layer 3
b4 = np.random.rand(num_of_classes, 1) / 240  # initialize bias for output layer
w4 = np.random.rand(num_of_classes, 240) / 240  # initialize weights for output layer

epochs = 5
m = 64  # batch size
n = 0.00001  # learning rate

TrMSE = np.zeros((epochs, 1))  # Training Mean Squared Error
VaMSE = np.zeros((epochs, 1))  # Validation Mean Squared Error

TrCA = np.zeros((epochs, 1))  # Training Mean Classification Accuracy
VaCA = np.zeros((epochs, 1))  # Validation Classification Accuracy

# print('batch size =', m)
# print('epochs =', epochs )

for epoch in range(0, epochs):
    a = time.time()
    for i in range(0, int(np.ceil(len(trainLabels) / m))):
        x = trainImages[(i * m):(i + 1) * m, :]
        z1 = b1.T + x.dot(w1.T)
        ## tanh ##
        # h1 = np.tanh(z1)  # calculate hidden 1 output
        # z2 = b2.T + h1.dot(w2.T)
        # h2 = np.tanh(z2)  # calculate hidden 2 output
        # z3 = b3.T + h2.dot(w3.T)
        # h3 = np.tanh(z3)  # calculate hidden 3 output
        ## tanh ##
        ## ReLu ##
        h1 = z1
        h1[z1 < 0] = 0
        z2 = b2.T + h1.dot(w2.T)
        h2 = z2
        h2[z2 < 0] = 0
        z3 = b3.T + h2.dot(w3.T)
        h3 = z3
        h3[z3 < 0] = 0
        ## ReLu ##
        z4 = b4.T + h3.dot(w4.T)
        yc = np.exp(z4)
        yc /= np.sum(yc, axis=1).reshape((x.shape[0], 1))  # calculate output
        yt = trainLabels[(i * m):((i + 1) * m)]  # known true output

        # Backpropagation
        # print('yt', yt.shape)
        # print('yc', yc.shape)
        error = (yt - yc) ** 2 / x.shape[0]
        # print('error', error.shape)
        gb4 = error.T  # gradient of bias of output layer
        # print('b4', b4.shape)
        # print('gb4', gb4.shape)
        gw4 = error.T.dot(h3)  # gradient of weights of output layer
        # print('w4', w4.shape)
        # print('gw4', gw4.shape)

        # print('w4', w4.shape)
        # print('z3', z3.shape)
        delta3 = error.dot(w4)  # * (1 - h3 ** 2)
        delta3[h3 < 0] = 0
        # print('delta3', delta3.shape)
        gb3 = delta3.T  # gradient of bias of hidden layer 3
        # print('b3', b3.shape)
        # print('gb3', gb3.shape)
        gw3 = gb3.dot(h2)  # gradient of weights of hidden layer 3
        # print('w3', w3.shape)
        # print('gw3', gw3.shape)

        # print('w3', w3.shape)
        # print('z2', z2.shape)
        delta2 = delta3.dot(w3)  # * (1 - h2 ** 2)
        delta2[h2 < 0] = 0
        # print('delta2', delta2.shape)
        gb2 = delta2.T  # gradient of bias of hidden layer 2
        # print('b2', b2.shape)
        # print('gb2', gb2.shape)
        gw2 = gb2.dot(h1)  # gradient of weights of hidden layer 2
        # print('w2', w2.shape)
        # print('gw2', gw2.shape)

        delta1 = delta2.dot(w2)  # * (1 - h1 ** 2)
        delta1[h1 < 0] = 0
        # print('delta1', delta1.shape)
        gb1 = delta1.T  # gradient of bias of hidden layer 1
        # print('b1', b1.shape)
        # print('gb1', gb1.shape)
        gw1 = gb1.dot(x)  # gradient of weights of hidden layer 1
        # print('w1', w1.shape)
        # print('gw1', gw1.shape)

        # # debugging
        # if np.sum(gw4) < 1 or np.sum(gb4) < 1 or np.sum(gw3) < 1 or np.sum(gb3) < 1 or np.sum(gw2) < 1 or np.sum(gb2) < 1 or np.sum(gw1) < 1 or np.sum(gb1) < 1:
        #     print('gw4', np.sum(gw4))
        #     print('gb4', np.sum(gb4))
        #     print('gw3', np.sum(gw3))
        #     print('gb3', np.sum(gb3))
        #     print('gw2', np.sum(gw2))
        #     print('gb2', np.sum(gb2))
        #     print('gw1', np.sum(gw1))
        #     print('gb1', np.sum(gb1))

        # Update weights
        w4 = w4 - n * gw4
        b4 = b4 - n * np.sum(gb4, 1).reshape(b4.shape)

        w3 = w3 - n * gw3
        b3 = b3 - n * np.sum(gb3, 1).reshape(b3.shape)

        w2 = w2 - n * gw2
        b2 = b2 - n * np.sum(gb2, 1).reshape(b2.shape)

        w1 = w1 - n * gw1
        b1 = b1 - n * np.sum(gb1, 1).reshape(b1.shape)

    z1 = b1.T + trainImages.dot(w1.T)
    ## tanh ##
    # h1 = np.tanh(z1)  # calculate hidden 1 output
    # z2 = b2.T + h1.dot(w2.T)
    # h2 = np.tanh(z2)  # calculate hidden 2 output
    # z3 = b3.T + h2.dot(w3.T)
    # h3 = np.tanh(z3)  # calculate hidden 3 output
    ## tanh ##
    ## ReLu ##
    h1 = z1
    h1[z1 < 0] = 0
    z2 = b2.T + h1.dot(w2.T)
    h2 = z2
    h2[z2 < 0] = 0
    z3 = b3.T + h2.dot(w3.T)
    h3 = z3
    h3[z3 < 0] = 0
    ## ReLu ##
    z4 = b4.T + h3.dot(w4.T)
    yc = np.exp(z4)
    yc /= np.sum(yc, axis=1).reshape((trainImages.shape[0], 1))  # calculate output
    TrMSE[epoch] += np.sum((trainLabels - yc) ** 2) / (2 * len(trainLabels))  # Training Squared Error
    TrCA[epoch] += 100 * np.sum(np.argmax(yc, axis=1) == np.argmax(trainLabels, axis=1)) / len(trainLabels)

    z1 = b1.T + valImages.dot(w1.T)
    ## tanh ##
    # h1 = np.tanh(z1)  # calculate hidden 1 output
    # z2 = b2.T + h1.dot(w2.T)
    # h2 = np.tanh(z2)  # calculate hidden 2 output
    # z3 = b3.T + h2.dot(w3.T)
    # h3 = np.tanh(z3)  # calculate hidden 3 output
    ## tanh ##
    ## ReLu ##
    h1 = z1
    h1[z1 < 0] = 0
    z2 = b2.T + h1.dot(w2.T)
    h2 = z2
    h2[z2 < 0] = 0
    z3 = b3.T + h2.dot(w3.T)
    h3 = z3
    h3[z3 < 0] = 0
    ## ReLu ##
    z4 = b4.T + h3.dot(w4.T)
    yc = (np.exp(z4, dtype='float64'))
    yc /= np.sum(yc, axis=1).reshape((valImages.shape[0], 1))  # calculate output
    VaMSE[epoch] = np.sum((valLabels - yc) ** 2) / (2 * len(valLabels))  # Validation Squared Error
    VaCA[epoch] = 100 * np.sum(np.argmax(yc, axis=1) == np.argmax(valLabels, axis=1)) / len(valLabels)

    # # Save model
    # with open('./' + str(epoch) + ' NN.npy', 'wb') as f:
    #     np.save(f, b1)
    #     np.save(f, w1)
    #     np.save(f, b2)
    #     np.save(f, w2)
    #     np.save(f, b3)
    #     np.save(f, w3)
    #     np.save(f, b4)
    #     np.save(f, w4)
    #     np.save(f, TrMSE)
    #     np.save(f, VaMSE)
    #     np.save(f, TrCA)
    #     np.save(f, VaCA)

    aa = time.time() - a
    if aa > 3600:
        print("epoch", epoch + 1, "took", '%.2f' % (aa / 3600), "hour(s)")
    elif aa > 60:
        print("epoch", epoch + 1, "took", '%.2f' % (aa / 60), "minutes")
    else:
        print("epoch", epoch + 1, "took", '%.2f' % aa, "seconds")
    aa = (epochs - epoch + 1) * aa
    if aa > 3600:
        print("ETA", '%.2f' % (aa / 3600), "hour(s)")
    elif aa > 60:
        print("ETA", '%.2f' % (aa / 60), "minutes")
    else:
        print("ETA", '%.2f' % aa, "seconds")

pyplot.figure()
pyplot.plot(TrMSE)
pyplot.title("Training")
pyplot.ylabel("Mean Squared Error")
pyplot.xlabel('epoch')
pyplot.savefig("./NN TrMSE.png")
pyplot.figure()
pyplot.plot(TrCA)
pyplot.title("Training")
pyplot.ylabel("Classification Accuracy")
pyplot.xlabel('epoch')
pyplot.savefig("./NN TrCA.png")
pyplot.figure()
pyplot.plot(VaMSE)
pyplot.title("Validation")
pyplot.ylabel("Mean Squared Error")
pyplot.xlabel('epoch')
pyplot.savefig("./NN VaMSE.png")
pyplot.figure()
pyplot.plot(VaCA)
pyplot.title("Validation")
pyplot.ylabel("Classification Accuracy")
pyplot.xlabel('epoch')
pyplot.savefig("./NN VaCA.png")

# with open('./' + str(epochs) + 'NN.npy', 'rb') as f:
#     b1 = np.load(f)
#     w1 = np.load(f)
#     b2 = np.load(f)
#     w2 = np.load(f)
#     b3 = np.load(f)
#     w3 = np.load(f)
#     b4 = np.load(f)
#     w4 = np.load(f)
#     TrMSE = np.load(f)
#     VaMSE = np.load(f)
#     TrCA = np.load(f)
#     VaCA = np.load(f)

del trainImages
del trainLabels
del valImages
del valLabels

# Load Test Data

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
    if i != num_of_classes:
        raise NameError('Number of classes are not same as training data')
else:
    testData = np.genfromtxt('./mnist_test.csv', delimiter=',')[1:]
    testImages = np.array(testData[0: (9 * testData.shape[0] // 10), 1:])
    testLabels = np.array(testData[0: (9 * testData.shape[0] // 10), 0], dtype=int)
    testImages = testImages.reshape((testImages.shape[0], 28, 28, 1))
    del testData

testImages = testImages.reshape((testImages.shape[0], -1))  # vectorize each observation

testImages = testImages / 255  # normalize # needs lots of RAM

testLabels = np.eye(num_of_classes)[testLabels]

z1 = b1.T + testImages.dot(w1.T)
## tanh ##
# h1 = np.tanh(z1)  # calculate hidden 1 output
# z2 = b2.T + h1.dot(w2.T)
# h2 = np.tanh(z2)  # calculate hidden 2 output
# z3 = b3.T + h2.dot(w3.T)
# h3 = np.tanh(z3)  # calculate hidden 3 output
## tanh ##
## ReLu ##
h1 = z1
h1[z1 < 0] = 0
z2 = b2.T + h1.dot(w2.T)
h2 = z2
h2[z2 < 0] = 0
z3 = b3.T + h2.dot(w3.T)
h3 = z3
h3[z3 < 0] = 0
## ReLu ##
z4 = b4.T + h3.dot(w4.T)
yc = np.exp(z4)
yc /= np.sum(yc, axis=1).reshape((testImages.shape[0], 1))  # calculate output
TeMSE = np.sum((testLabels - yc) ** 2) / (2 * len(testLabels))  # Testing Squared Error
TeCA = 100 * np.sum(np.argmax(yc, axis=1) == np.argmax(testLabels, axis=1)) / len(testLabels)

print("Test Mean Squared Error", TeMSE)
print("Test Classification Accuracy", TeCA, "%")

pyplot.figure()
pyplot.imshow(confusion_matrix(np.argmax(yc, axis=1), np.argmax(testLabels, axis=1)))
pyplot.savefig("./NN cm.png")

aa = time.time() - blah
if aa > 3600:
    print("Time Elapsed to run whole code", '%.2f' % (aa / 3600), "hour(s)")
elif aa > 60:
    print("Time Elapsed to run whole code", '%.2f' % (aa / 60), "minutes")
else:
    print("Time Elapsed to run whole code", '%.2f' % aa, "seconds")
