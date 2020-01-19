import scipy.io
import numpy as np


#naive bayes classifier for returning the best class
def naiveBayesClassifier(x1, x2, mu1_7, std1_7, mu2_7, std2_7, mu1_8, std1_8, mu2_8, std2_8, preProb7, preProb8):
	norm1_7 = (1/(np.sqrt(2 * np.pi * (std1_7 ** 2)))) * np.exp(-((x1 - mu1_7) ** 2)/(2 * (std1_7 ** 2)))
	norm2_7 = (1/(np.sqrt(2 * np.pi * (std2_7 ** 2)))) * np.exp(-((x2 - mu2_7) ** 2)/(2 * (std2_7 ** 2)))

	norm1_8 = (1/(np.sqrt(2 * np.pi * (std1_8 ** 2)))) * np.exp(-((x1 - mu1_8) ** 2)/(2 * (std1_8 ** 2)))
	norm2_8 = (1/(np.sqrt(2 * np.pi * (std2_8 ** 2)))) * np.exp(-((x2 - mu2_8) ** 2)/(2 * (std2_8 ** 2)))

	probability1_7 = norm1_7 * preProb7
	probability2_7 = norm2_7 * preProb7
	probability7 = probability1_7 * probability2_7

	probability1_8 = norm1_8 * preProb8
	probability2_8 = norm2_8 * preProb8
	probability8 = probability1_8 * probability2_8

	if probability7 > probability8:
		return 0
	else:
		return 1


#Starting main
#loading dataset
Numpyfile= scipy.io.loadmat('data/mnist_data.mat')

#separating training and testing data and labels
trainX = Numpyfile['trX']
trainY = Numpyfile['trY']
testX = Numpyfile['tsX']
testY = Numpyfile['tsY']

#separating data for digit 7 and 8
trainX7 = []
trainX8 = []
testX7 = []
testX8 = []
for i in range(len(trainY[0])):
	if trainY[0][i] == 0:
		trainX7.append(trainX[i])
	else:
		trainX8.append(trainX[i])
for i in range(len(testY[0])):
	if testY[0][i] == 0:
		testX7.append(testX[i])
	else:
		testX8.append(testX[i])

#Converting to numpy array
trainX7 = np.asarray(trainX7)
trainX8 = np.asarray(trainX8)
testX7 = np.asarray(testX7)
testX8 = np.asarray(testX8)

#Calculating features for training data
meanX7 = []
meanX8 = []
sdX7 = []
sdX8 = []
for i in range(len(trainX7)):
	meanX7.append(trainX7[i].mean())
	sdX7.append(trainX7[i].std())
for i in range(len(trainX8)):
	meanX8.append(trainX8[i].mean())
	sdX8.append(trainX8[i].std())
meanX7 = np.asarray(meanX7)
meanX8 = np.asarray(meanX8)
sdX7 = np.asarray(sdX7)
sdX8 = np.asarray(sdX8)

#prior probabilities
preProb7 = len(trainX7)/(len(trainX7) + len(trainX8))
preProb8 = len(trainX8)/(len(trainX7) + len(trainX8))

#calculating parameters for 2d normal distribution
#parameters for digit 7
mu1_7 = meanX7.mean()
std1_7 = meanX7.std()
mu2_7 = sdX7.mean()
std2_7 = sdX7.std()

#parameters for digit 8
mu1_8 = meanX8.mean()
std1_8 = meanX8.std()
mu2_8 = sdX8.mean()
std2_8 = sdX8.std()

#predicting class for test data digit 7
predictedY7 = []
for testData in testX7:
	x1 = testData.mean()
	x2 = testData.std()
	predictedClass = naiveBayesClassifier(x1, x2, mu1_7, std1_7, mu2_7, std2_7, mu1_8, std1_8, mu2_8, std2_8, preProb7, preProb8)
	predictedY7.append(predictedClass)

#predicting class for test data digit 7
predictedY8 = []
for testData in testX8:
	x1 = testData.mean()
	x2 = testData.std()
	predictedClass = naiveBayesClassifier(x1, x2, mu1_7, std1_7, mu2_7, std2_7, mu1_8, std1_8, mu2_8, std2_8, preProb7, preProb8)
	predictedY8.append(predictedClass)

#calculating accuracies
nb_correct7 = 0
for value in predictedY7:
	if value == 0:
		nb_correct7 += 1
nb_correct8 = 0
for value in predictedY8:
	if value == 1:
		nb_correct8 += 1
nb_accuracy7 = nb_correct7/len(predictedY7)
nb_accuracy8 = nb_correct8/len(predictedY8)
nb_accuracy = (nb_correct7 + nb_correct8)/(len(predictedY7) + len(predictedY8))
print("Accuracy of Digit 7 in Naive Bayes Classifier: " + str(round(nb_accuracy7 * 100, 2)) + "%")
print("Accuracy of Digit 8 in Naive Bayes Classifier: " + str(round(nb_accuracy8 * 100, 2)) + "%")
print("Overall Accuracy in Naive Bayes Classifier: " + str(round(nb_accuracy * 100, 2)) + "%")





