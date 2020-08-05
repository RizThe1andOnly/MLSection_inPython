import numpy as np
import sklearn.linear_model
import math


NUMBER_OF_SAMPLES = 100
NUMBER_OF_FEATURES = 7

#for now will create random data with which to carry out operations:
def createRandomData():
    #generate gaussian random value array
    x = np.random.normal(0,1,(NUMBER_OF_SAMPLES,NUMBER_OF_FEATURES))
    return x

#run sci-kit learn logistic regression code:
def runMultiNomialTraining(inputArray):
    #delete the pass statement below and put sklearn code below
    #link to specific pages we need right now: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    # and : https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    pass


#start of Iterative Reweighted Least Squares training algo; will have all of the sub-functions

#obtaining fitted probabilities:

def getExp_of_linearFunction(Bk,xi):
    """
        Will obtain: exp(Bk0 + (Bk)T * xi)

        Bk = Row of coefficients; Bk0 is the first coeff; k corresponds to which outcome is being tested
        xi = a sample
    """
    bk0 = Bk[0]
    bk = np.array(Bk[:])
    ans = bk0 + np.dot(bk,xi)
    expAns = math.exp(ans)
    return expAns

def getExp_of_linerFuntion_Summation(B,x,K):
    """
        Will obtain: sum(1,k-1,exp(Bk0+(Bk)T * xi))

        B = matrix of coeffs
        x = feature vector or a sample row
        K = number of features + 1. Note indices start from 0
    """
    summation = 0
    for i in range(K):
        summation = summation + getExp_of_linearFunction(B[i],x)
    return summation

def getProbabilityOf_outcome_given_features(B,xi,K,k):
    """
        Will obtain: ( exp(Bk0+(Bk)T*xi) / 1 + sum(1,k-1,exp(Bk0+(Bk)T*xi)) )

        Outcomes are: k = 0...K-1

        B = matrix of coeffs

        xi = features input or a sample row

        K = number of features + 1 ?

        k = which outcome in particular
    """
    numerator = getExp_of_linearFunction(B[k],xi)
    denominator = 1 + getExp_of_linerFuntion_Summation(B,xi,K)
    ans =  numerator / denominator
    return ans

def getFittedProbabilityVector(B,X,K):
    """
        Obtains the vector of fitted probabilities where the ith elem is 
        Pr(Y=k|X=xi;B)
    """


