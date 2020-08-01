import numpy as np
import sklearn.linear_model


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


randomArr = createRandomData()
#runMultiNomialTraining(randomArr) #un-comment this command once runMultinomialTraining has been written
