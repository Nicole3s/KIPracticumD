# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Mira implementation
import util
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        "*** YOUR CODE HERE ***"
        #go through every value in C
        for C in Cgrid:
        #notate which C values you are using (iteration)
            print "Going through Cvalue ", C, " ..."
        #go through all the values in trainingData
            for i in range(len(trainingData)):
                #sort all the values in the trainingData
                labels = self.classify([trainingData[i]]) #y'
                #take the highest value
                maxscore = labels[0]
                #if the highest value is not equal to the trainingLabel value, continue, else, do nothing
                
                if maxscore != trainingLabels[i]:
                    print trainingLabels[i]
                    
                    #define f normalized
                    normalizedData = trainingData[i].copy()
                    if normalizedData != None:
                        normalizedData = util.Counter.normalize(normalizedData)

                    #calculate step size: stepsize = ((wy' - wy)*f + 1)/2(f norm)^2
                    stepSize = (((self.weights[maxscore] - self.weights[trainingLabels[i]])*trainingData[i]) + 1)/(2*(normalizedData*normalizedData))
                    #take the minimum of the step size and the C value used
                    stepSizeFinal = min(C, stepSize)
                    # https://github.com/anthony-niklas/cs188/blob/master/p5/mira.py
                    data = trainingData[i].copy()
                    #calculate f*stepsize
                    data.divideAll(1/stepSizeFinal)
                    # calculate the value of the weight of y: wy = wy + f*stepsize
                    self.weights[trainingLabels[i]] += data  # y
                    # calculate the value of the weight of y': wy' = wy' - f*stepsize
                    self.weights[maxscore] -= data # y'

    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses


