from mnist import MNIST
import numpy as np
import math
from numpy.linalg import inv
from numpy.linalg import det
from scipy.stats import multivariate_normal
mndata = MNIST('samples')

trainX, trainY = mndata.load_training()
#trainX is 60000 rows, 784 columns
#trainY is the corresponding labels with 60000 rows

testX, testY = mndata.load_testing()
#testX is 10000 rows, 784 columns
#testY is the corresponding labels with 10000 rows

trainX = np.array(trainX)
trainY = np.array(trainY)
testX = np.array(testX)
testY = np.array(testY)

training_indices = range(2500,15000) + range(17500, 30000) + range(32500, 45000) + range(47500,60000)
validation_indices = range(0,2500) + range(15000,17500) + range(30000,32500) + range(45000, 47500)
trainx = trainX[training_indices, :]
trainy = trainY[training_indices]
validx = trainX[validation_indices, :]
validy = trainY[validation_indices]

labels = [None] * 10
for i in range(10):
    labels[i] = []

#parse label 0-9 for training set
for i in range(len(trainy)):
    labels[trainy[i]].append(trainx[i])

labels = np.array(labels)

#calculate mean
means = np.array([None] * 10)
for i in range(10):
    means[i] = np.mean(labels[i],axis=0)

#calculate covariance matrix
covs = np.array([None] * 10)
for i in range(10):
    covs[i] = np.cov(labels[i],rowvar=0,ddof=0)

#vector of c
c = [0.01,0.1,1,10,100,500,1000,2000,3000,4000,5000,6000,7000,8000]

#calculate density function
def density_function(x,mean,cov,cofactor):
    smooth_matrix = np.matrix(cov) + cofactor*np.identity(len(cov))
    inverse_matrix = inv(cov)
    distance_to_mean = np.array(x)-np.array(mean)
    quadratic_function = np.matmul(np.matmul(distance_to_mean.T, inverse_matrix),distance_to_mean)
    determinant = det(cov)
    density = ( 1 / math.pow((2*math.pi),len(x)/2) ) * math.exp(-1/2 * quadratic_function)
    return density

def smooth_matrix(covs, cofactor, smoothed_matrix):
    for i in range(len(covs)):
        smoothed_matrix[i] = covs[i] + np.eye(covs[i].shape[1])*cofactor

def multivariate_gaussian(results, testX, testY, p, misclassified=None):
    err = 0
    for i in range(len(testX)):
        densities = np.array([None] * 10)
        for j in range(len(densities)):
            densities[j] = p[j].logpdf(testX[i])
        label = np.argmax(densities)
        results[i] = label
        if(results[i]!=testY[i]):
            err += 1
            if(misclassified != None):
                misclassified.append(i)
    return float(err)/float(len(testY))

def find_best_cofactor(c):
    err_rates = np.array([None] * len(c))
    for i in range(len(c)):
        #cross validate
        validations = np.array([None] * len(validy))
        smoothed_matrix = np.array([None] * 10)
        smooth_matrix(covs,c[i],smoothed_matrix)
        p = np.array([None] * 10)
        for j in range(len(p)):
            p[j] = multivariate_normal(means[j],smoothed_matrix[j])
        err_rate = multivariate_gaussian(validations, validx, validy,p)
        err_rates[i] = err_rate
    min_index = np.argmin(err_rates)
    print(err_rates)
    return min_index


#best_c = find_best_cofactor(c)
#print(best_c)
results = [None] * len(testY)
smoothed_matrix = np.array([None] * 10)
smooth_matrix(covs,c[9],smoothed_matrix)
p = np.array([None] * 10)
for j in range(len(p)):
    p[j] = multivariate_normal(means[j],smoothed_matrix[j])
misclassified = []
err_rate = multivariate_gaussian(results, testX, testY,p , misclassified)
print(err_rate)
top5misclassfied = misclassified[0:5]
for i in range(len(top5misclassfied)):
    print(mndata.display(testX[top5misclassfied[i]]))
    print(results[top5misclassfied[i]])
    densities = np.array([None] * 10)
    probX = 0;
    for j in range(len(densities)):
        densities[j] = p[j].logpdf(testX[top5misclassfied[i]])
        probX += densities[j]
    for k in range(len(densities)):
        print(float(densities[k]) / float(probX))
