# -*- coding: utf-8 -*-
# Flora Seo 
# 2020/9/27
# Frequentist Machine Learning_Project2

# Data
# The data are taken from a larger dataset, described in Rousseauw et al.1983, South African Medical Journal, 
# featuring the retrospective sample of males in a heart-disease high-risk region of the Western Cape, South Africa.
# Data from : https://web.stanford.edu/~hastie/ElemStatLearn/datasets/SAheart.data

# Set up
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn import preprocessing
import seaborn as sb
from functools import partial

# Base Logistic Classifier
# All logistic classifiers subclass from this. 
# For all 3 models (unregularized, stepwise, and L2 regularized) 
# Base logistic classifier class; includes some common utilities;
# This class is abstract and doesn't contain a train() or validate() method,
# Which must be implemented in its inheritors
# Base Logistic Classifier
class BaseLogisticClassifier:

    # X: NxP ndarray (features)
    # y: Nx1 ndarray (labels)
    # Alpha: (maximum) learning rate
    # copySubsetsFrom: treat this as a copy constructor, copy
    # over the subsets from a different BaseLogisticClassifier
    def __init__(self, X, y, alpha=0.01,
                 copySubsetsFrom=None,
                 standardizeFeatures=False):
        self._alpha = alpha
        self._lambda = 0

        # copySubsetsFrom is provided; copy subsplits
        if copySubsetsFrom is not None:
            self._subsets = copySubsetsFrom._subsets.copy()
            P = self._subsets['train']['X'].shape[1] - 1

        # X, y are provided; manually split subsets
        else:
            N, P = X.shape

            # Add column of 1's to X
            X = np.hstack((np.ones((N, 1)), X))

            # Random data split into training, validation, test
            indices = np.arange(N)
            np.random.shuffle(indices)
            split1, split2 = int(N*0.8), int(N*0.9)
            self._subsets = {
                'train': {
                    'X': X[indices[:split1], :],
                    'y': y[indices[:split1], :]
                },
                'validate': {
                    'X': X[indices[split1:split2], :],
                    'y': y[indices[split1:split2], :]
                },
                'test': {
                    'X': X[indices[split2:], :],
                    'y': y[indices[split2:], :]
                }
            }

            # Check the lengths of the dataset and each set
            print("Length of dataset:", N)
            print("Length of training:", split1)
            print("Length of validation:", split2-split1)
            print("Length of test:", N-split2)

        # Initialize weight vector including Bias : P+1
        self._theta = np.zeros((P+1, 1))


        # epsilon**
        # Prevent any division by zero in the implementation 
        self._ep = 0.0001

        # Standardize features except for first column of 1's
        if standardizeFeatures:
            for subset in ('train', 'validate', 'test'):
                subsetX = self._subsets[subset]['X']
                # print(subsetX[:,1,np.newaxis])
                self._subsets[subset]['X'] = np.hstack((subsetX[:,1,np.newaxis], 
                                                        preprocessing.StandardScaler()
                                                        .fit(subsetX[:,1:])
                                                        .transform(subsetX[:,1:])))
            
    def theta(self):
        return self._theta
        
# Binary Logistic Classifier
# This is the unregularized case for the binary logistic classifier. 
# It uses an Stochastic Gradient Descent optimization step() function according to the problem statement.
# Notes on implementation:
# SGD as mentione in Problem Statement
# The functions are all vectorized and the batch size is arbitrary. Right now, train() uses the entire training dataset on each iteration (full batch).
# Binary Logistic Classifier
class BinaryLogisticClassifier(BaseLogisticClassifier):

    # Hypothesis function (returns yhat); uses trained theta
    # Returns N x 1
    def h(self, X):
        return 1 / (1 + np.exp(-X @ self._theta))

    def grad(self, X, y):
        return X.T @ (y - self.h(X))

    # Percent classified wrong on training subset
    def pctWrong(self, subset='test'):
        X, y = self._subsets[subset]['X'], self._subsets[subset]['y']
        N, _ = X.shape

        # epsilon to prevent prediction of exactly 0.5 to be classified as
        # correct for label being either 0 or 1
        ep = 0.00001

        return np.sum(np.round(np.abs(self.h(X) - y - ep))) / N

    # SGD update step
    def step(self, iter, includeMask=None):
        
        thetaGrad = self.grad(self._subsets['train']['X'], self._subsets['train']['y'])

        self._theta = self._theta + self._alpha*thetaGrad
        # exclude certain features (for stepwise)
        if includeMask is not None:
            self._theta *= includeMask
            

    def train(self, iterations=2000, includeMask=None):
        self._theta = np.zeros((self._subsets['train']['X'].shape[1], 1))

        for i in range(iterations):
            self.step(i, includeMask)


    # Baseline sets the bias to the average label, and zeros elsewhere,
    # and calculates the percent wrong on the test dataset
    def baselinePctWrong(self):
        X, y = self._subsets['test']['X'], self._subsets['test']['y']
        N, _ = X.shape
        return np.sum(np.round(np.abs(np.average(self._subsets['train']['y']) - y))) / N
     
# Stepwise Logistic Classifier
# This inherits from the binary logistic classifier, and adds a validate function that follows the algorithm:
class StepwiseLogisticClassifier(BinaryLogisticClassifier):

    def validate(self):
        _, P = self._subsets['train']['X'].shape
        P -= 1

        # list of features to exclude and include
        exclude = list(range(P))
        include = []
        # list of features to include
        includeMask = np.zeros((P+1, 1)) 
        includeMask[0] = 1

        pctWrongs = np.zeros((P+1, 1))
        #calculate the percent wrong relative to the validate set of data
        pctWrongs[0] = 1 - np.mean(self._subsets['validate']['y'])

        # loops over number of features in model
        for i in range(P):

            # find best next feature to include
            bestPctWrong, bestFeature = float('inf'), None
            for feature in exclude:
                # copy includeMask into currentIncludeMask, unmask feature
                currentIncludeMask = np.array(includeMask)
                currentIncludeMask[feature+1] = 1

                # train on currentIncludeMask
                self.train(includeMask=currentIncludeMask)

                # calculate percent wrong on validation set
                #Trying to find out when it gives you the least error
                pctWrong = self.pctWrong(subset='validate')
                if pctWrong < bestPctWrong:
                    bestPctWrong = pctWrong
                    bestFeature = feature

            # minimize percent wrong
            pctWrongs[i+1] = bestPctWrong

            # add feature to includeMask, remove from exclude
            exclude.remove(bestFeature)
            include.append(bestFeature)
            includeMask[bestFeature] = 1

        # find minimum of pctWrongs
        bestNumFeatures = np.argwhere(pctWrongs == np.min(pctWrongs))[0,0]
        bestIncludeMask = np.zeros((P+1, 1))
        bestIncludeMask[0] = 1
        for i in range(bestNumFeatures):
            bestIncludeMask[include[i]+1] = 1

        # retrain with best include mask, return theta
        self.train(includeMask=bestIncludeMask)
        return self._theta, include[:bestNumFeatures]
        
# L2 Logistic Classifier
# This inherits from the binary logistic classifier, and modifies the gradient to penalize the bias.
#Class to calculate L2 regularization
class L2LogisticClassifier(BinaryLogisticClassifier):

    # make sure to standardize features
    def __init__(self, X, Y, alpha=0.01, copySubsetsFrom=None):
        super().__init__(X, Y, alpha=alpha,
                         copySubsetsFrom=copySubsetsFrom,
                         standardizeFeatures=True)

    # update function with L2 penalty
    # theta_j := theta_j + alpha(y_i -h_theta(x_i)) * x_i_j
    # returns (P+1)
    # SGD =  j+α(y(i)−hθ(x(i)))x(i)j
    def grad(self, X, y):
        # don't penalize the bias
        return X.T @ (y - self.h(X)) - 2 * self._lambda * np.vstack((np.zeros((1,1)), self._theta[1:,:]))

    def validate(self):
        #create a bunch of lambdas in order to iterate through them
        lams = np.logspace(-20, 5, 100)

        #Removing the ones because we don't want to regularize the bias term
        P = self._subsets['train']['X'].shape[1] - 1
        self._subsets['train']['X'][0,:] = np.ones((1, P+1))
        self._subsets['validate']['X'][0,:] = np.ones((1, P+1))
        self._subsets['test']['X'][0,:] = np.ones((1, P+1))

        bestPctWrong, bestLambda = float('inf'), None
        pctWrongs = np.zeros_like(lams)

        for i, lam in enumerate(lams):
            self._lambda = lam
            self.train()

            pctWrong = self.pctWrong(subset='validate')
            pctWrongs[i] = pctWrong
            # calculate percent wrong on validation set
            #Trying to find out when it gives you the least error
            if pctWrong < bestPctWrong:
                bestPctWrong = pctWrong
                bestLambda = lam

        self._lambda = bestLambda
        self.train()
        return self._theta

# L1 Logistic Classifier
# taken mostly literally from (Tsuruoka et al., 2009); involves an
# estimate of the gradient of the L1 norm (abs function) that involves
# some "memory" for improved performance
class L1LogisticClassifier(BinaryLogisticClassifier):

    # make sure to standardize features
    def __init__(self, X, Y, alpha=0.01, copySubsetsFrom=None):
        super().__init__(X, Y, alpha=alpha,
                         copySubsetsFrom=copySubsetsFrom,
                         standardizeFeatures=True)

    
    def applyL1Penalty(self):
        for i, theta_i in enumerate(self._theta.reshape(-1)):
            # start from 1 to not penalize the bias
            if i == 0:
                continue

            z = theta_i
            if theta_i > 0:
                self._theta[i,0] = max(0., theta_i - (self._u + self._q[i]))
            elif theta_i < 0:
                self._theta[i,0] = min(0., theta_i + (self._u + self._q[i]))
            self._q[i] += theta_i - z

    # log likelihood
    def l(self, subset):
        X, y = self._subsets[subset]['X'], self._subsets[subset]['y']
        return y.T @ np.log(self.h(X)) + (1 - y).T @ np.log(1 - self.h(X))

    def train(self, iterations):
        self._theta *= 0.
        self._u = 0.
        self._q = self._theta.copy().reshape(-1)
        self._N = self._subsets['train']['X'].shape[0]

    

        for i in range(iterations):
            self._u += self._alpha * self._C / self._N
            self.step(i)
            self.applyL1Penalty()


    def validate(self, iterations=2000):
        # just to be sure; undo l2 regularization
        self._lambda = 0.

        # l1 regularization parameter; C is the letter used in the text
        cIteration = np.logspace(-8, 0, 30)
        bestPctWrong, bestC = float('inf'), None
        pctWrongs = np.zeros_like(cIteration)

        # note: coefficients includes bias
        coefficients = np.zeros((cIteration.size, self._theta.size))

        for j, c in enumerate(cIteration):
            self._C = c
            self.train(iterations)

            coefficients[j,:] = self._theta.reshape(-1)
            pctWrong = self.pctWrong(subset='validate')
            pctWrongs[j] = pctWrong
            if pctWrong < bestPctWrong:
                bestPctWrong = pctWrong
                bestC = c

                bestTheta = self._theta.copy()
                
        self._C = bestC

        self._theta = bestTheta

        return self._C, coefficients
 
# Multinomial (Trinary) Logistic Regression
# multinomial case, uses simple SGD
class MultinomialLogisticClassifier(BaseLogisticClassifier):

    # returns NxK matrix, where each row is the predicted probabilities
    # of each of the K classes
    def h(self, X):
        return 1 / (1 + np.exp(-X @ self._theta))

    
    def grad(self , X , y):
        return X.T @ (y - self.h(X))


    # calculate percent wrong: compares argmax of estimate and label
    def pctWrong(self, subset='test'):
        X, y = self._subsets['train']['X'], self._subsets['train']['y']
        N = X.shape[0]
        return np.sum(np.round(np.abs(np.argmax(self.h(X), axis=1) - \
            np.argmax(y, axis=1)))) / N

    # hardcoded 3-class classifier (e.g., for UCI Iris dataset)
    def MultinomialClassificationTrain(self, iterations=2000):
        N, P = self._subsets['train']['X'].shape
        P -= 1
        Q = self._subsets['train']['y'].shape[1]
        X, y = self._subsets['train']['X'], self._subsets['train']['y']
        
        self._theta = np.random.rand( P+1 , Q)
        
        for i in range(iterations):
            # use basic sgd 
            grads = self.grad(X,y)
            self._theta += self._alpha * grads

        return self._theta

# Running Model 
# SAHD 
# Import the South African heart disease dataset (in Google Colab)
# Read through data and create dataset
sahdDataset = pd.read_csv('https://web.stanford.edu/~hastie/ElemStatLearn/datasets/SAheart.data', index_col=0)

# Textbook drops adiposity and typea
sahdDataset = sahdDataset.drop(['adiposity', 'typea'], axis=1)
# Turn famhist into a quantitative variable
sahdDataset['famhist'] = (sahdDataset['famhist'] == 'Present')*1
# Creates a graph like figure 4.12
sb.pairplot(sahdDataset, hue = 'chd',palette="hls")
# list the features
term = list(sahdDataset.columns.values[:-1])
# Generate Features matrix : NxP
sahdDatasetX = sahdDataset.drop(['chd'], axis=1).values
# Generate Label matrix : Nx1
sahdDatasety = sahdDataset.loc[:, 'chd'].values.reshape(-1, 1)


# PART 1: RECREATE TABLE 4.
binaryClassifier = BinaryLogisticClassifier(sahdDatasetX, sahdDatasety)
binaryClassifier.train()
correct_unregularized = np.around((1 - binaryClassifier.pctWrong()) * 100)
print(f'theta: {binaryClassifier.theta()}\n% classified correct for unregularized: {np.around((1 - binaryClassifier.pctWrong()) * 100)}%')
correct_baseline = np.around((1. - binaryClassifier.baselinePctWrong()) * 100)

# PART 2: STEPWISE
stepwiseClassifier = StepwiseLogisticClassifier(None, None, copySubsetsFrom=binaryClassifier)
_, optimalFeatures = stepwiseClassifier.validate()
correct_stepwise = np.around((1 - stepwiseClassifier.pctWrong()) * 100)
print(f'theta: {stepwiseClassifier.theta()}\n% classified correct for stepwise: {np.around((1 - stepwiseClassifier.pctWrong()) * 100)}%')
#report which features are the most important
print('The most important features(s) in order: ', [term[optimalFeature] for optimalFeature in optimalFeatures])

# PART 3: L2 REGULARIZATION
l2Classifier = L2LogisticClassifier(None, None, copySubsetsFrom=binaryClassifier)
l2Classifier.validate()
correct_L2regularized = np.around((1 - l2Classifier.pctWrong()) * 100)
print(f'theta: {l2Classifier.theta()}\n% classified correct for L2 regularized: {np.around((1 - l2Classifier.pctWrong()) * 100)}%')

#  L1 REGULARIZATION
l1Classifier = L1LogisticClassifier(None, None, copySubsetsFrom=binaryClassifier)
bestC, coefficients = l1Classifier.validate()
cIterations = np.logspace(-8, 0, 30)
plt.figure()
plt.plot(cIterations, coefficients[:,1:] , label = ('1' , '2' , '3' , '4' , '5' , '6'))
plt.xlabel('theta')
plt.ylabel('Coefficients')
plt.title('Lasso Coefficients')
plt.legend()
correct_L1regularized = np.around((1 - l1Classifier.pctWrong()) * 100)
print(f'theta: {l1Classifier.theta()}\n% classified correct for L1 regularized: {np.around((1 - l1Classifier.pctWrong()) * 100)}%')

# Additionally, report the % correct for all 3 models (unregularized, stepwise, and L2 regularized) in a table.
correct = list([[correct_baseline, correct_unregularized, correct_stepwise, correct_L2regularized, correct_L1regularized]])
models = list(['Baseline', 'Unregularized', 'Stepwise', 'L2 regularization', 'L1 regularization'])
table = tabulate(correct, headers=models,tablefmt='pretty')
print('Table 1: % Correct for all the Models')
print(table)

# Additional Example_1
# Breast Cancer data from Wisconsin University 
# Dataset Description: http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.names 
# Read through data and create dataset
bcDataset = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', names=["ID", "ClumpThickness", "Uniformity_CellSize", "Uniformity_CellShape", "MarginalAdhesion",
                             "SingleEpithelialCellSize", "BareNuclei", "BlandChromatin", "NormalNucleoli", 
                             "Mitosis", "Class"])
bcDataset.pop('ID')

# There are ? for missing data, so we drop the rows that have them 
bcDataset = bcDataset.apply(partial(pd.to_numeric, errors='coerce'))
bcDataset = bcDataset.dropna(axis = 0)
# Change labels to have 0 for benign and 1 for malignant
bcDataset['Class'] = (bcDataset['Class'] == 4)*1
# Creates a graph like figure 4.12
sb.pairplot(bcDataset, hue = 'Class',palette="hls")
# list the features
term = list(bcDataset.columns.values[:-1])
# Generate Features matrix : NxP
bcDatasetX = bcDataset.drop(['Class'], axis=1).values
# Generate Label matrix : Nx1
bcDatasety = bcDataset.loc[:, 'Class'].values.reshape(-1, 1)

# Running Model
# Breast Cancer Wisconsin 
# PART 1: RECREATE TABLE 4.
binaryClassifier = BinaryLogisticClassifier(bcDatasetX, bcDatasety)
binaryClassifier.train()
correct_unregularized = np.around((1 - binaryClassifier.pctWrong()) * 100)
print(f'theta: {binaryClassifier.theta()}\n% classified correct for unregularized: {np.around((1 - binaryClassifier.pctWrong()) * 100)}%')
correct_baseline = np.around((1 - binaryClassifier.baselinePctWrong()) * 100)

# PART 2: STEPWISE
stepwiseClassifier = StepwiseLogisticClassifier(None, None, copySubsetsFrom=binaryClassifier)
_, optimalFeatures = stepwiseClassifier.validate()
correct_stepwise = np.around((1 - stepwiseClassifier.pctWrong()) * 100)
print(f'theta: {stepwiseClassifier.theta()}\n% classified correct for stepwise: {np.around((1 - stepwiseClassifier.pctWrong()) * 100)}%')
#report which features are the most important
print('The most important features(s) in order: ', [term[optimalFeature] for optimalFeature in optimalFeatures])

#PART 3: L2 REGULARIZATION
l2Classifier = L2LogisticClassifier(None, None, copySubsetsFrom=binaryClassifier)
l2Classifier.validate()
correct_L2regularized = np.around((1 - l2Classifier.pctWrong()) * 100)
print(f'theta: {l2Classifier.theta()}\n% classified correct for L2 regularized: {np.around((1 - l2Classifier.pctWrong()) * 100)}%')

# L1 REGULARIZATION
l1Classifier = L1LogisticClassifier(None, None, copySubsetsFrom=binaryClassifier)
bestC, coefficients = l1Classifier.validate()
cIterations = np.logspace(-8, 0, 30)
plt.figure()
plt.plot(cIterations, coefficients[:,1:])
plt.xlabel('theta')
plt.ylabel('Coefficients')
plt.title('Lasso Coefficients')
correct_L1regularized = np.around((1 - l1Classifier.pctWrong()) * 100)
print(f'theta: {l1Classifier.theta()}\n% classified correct for L1 regularized: {np.around((1 - l1Classifier.pctWrong()) * 100)}%')

# Additionally, report the % correct for all 3 models (unregularized, stepwise, and L2 regularized) in a table.
correct = list([[correct_baseline, correct_unregularized, correct_stepwise, correct_L2regularized, correct_L1regularized]])
models = list(['Baseline', 'Unregularized', 'Stepwise', 'L2 regularization', 'L1 regularization'])
table = tabulate(correct, headers=models,tablefmt='pretty')
print('Table 1: % Correct for all the Models')
print(table)


# Additional Example_2
# IRIS data
# iris dataset for multiclass (3-class regression)
irisDataset = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')

# one-hot encode labels
irisDatasety = np.vstack((
    (irisDataset.iloc[:,4] == 'Iris-setosa').values.astype(np.float32),
    (irisDataset.iloc[:,4] == 'Iris-versicolor').values.astype(np.float32),
    (irisDataset.iloc[:,4] == 'Iris-virginica').values.astype(np.float32))).T

# feature matrix
irisDatasetX = irisDataset.iloc[:,:4].values
irisClassifier = MultinomialLogisticClassifier(irisDatasetX, irisDatasety)
irisClassifier.MultinomialClassificationTrain()
print(f'% correct on iris dataset: {(1 - irisClassifier.pctWrong())*100}')
# pd.DataFrame(data=np.hstack((irisClassifier._theta1, irisClassifier._theta2)),
#              columns=['theta_1', 'theta_2'],
#              index=['bias', 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
print('Comparison of prediction to actual labels')
pd.DataFrame(data=np.hstack((np.around(irisClassifier.h(irisClassifier._subsets['test']['X']), 1), irisClassifier._subsets['test']['y'])),
             columns=['PC1', 'PC2', 'PC3', 'AC1', 'AC2', 'AC3'])
