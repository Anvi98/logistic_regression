import numpy as np
from module import *

## Model
class LogisticRegAlex:
    def __init__(self):
        self.alpha = 0.5
        self.history = {}

    def train(self, X_train, Y_train, iteration):
        self.X_train = X_train
        self.Y_train = Y_train
        self.iteration = iteration
        self.W = np.zeros(X_train.shape[1])
        self.W = self.W.reshape(1,-1)

        #Training
        for i in range(self.iteration):
            self.Z = np.dot(self.W, self.X_train.T)
            self.y_hat = sigmoid(self.Z)
            self.y_hat = self.y_hat.reshape(-1,1)

            ### compute error
            # We multiplied by 0.5 to make sure the model doesn't overfit
            self.cost = 0.5 * np.absolute(np.mean((np.dot(-Y_train, np.log(self.y_hat.T)) +  np.dot((1-self.Y_train), np.log(1-self.y_hat.T)))))

            #Print cost at i iterations:
            if i % 100 == 0 or i == 1000:
                # Save iterations and cost for plotting
                print(f'Cost: {self.cost}  at iteration {i}')
                self.history[i] = self.cost

            ## Compute gradient
            self.dJ_dW = 1/len(self.X_train)*np.dot((self.y_hat - self.Y_train).T, self.X_train)

            ## Update weight with gradient descent
            self.W = self.W - self.alpha * self.dJ_dW
        
        return self.W, self.history
    
    def test(self, X, Y):
        self.X = X
        self.Y = Y

        ### Predictions 
        self.z_test = np.dot(self.W, self.X.T)
        self.y_hat_test = sigmoid(self.z_test)

        # Threshold
        self.y_hat_test = [1 if p > 0.5 else 0 for i in self.y_hat_test for p in i ]

        # This will help us get a list of labels not a list of lists
        self.Y = [y for i in self.Y for y in i]
        
        # Confusion Matrix
        self.tp, self.fp, self.tn, self.fn = get_confusion_matrix(self.Y, self.y_hat_test)

        ## Accuracy
        self.acc = (self.tp + self.tn)/ (self.tp+ self.tn + self.fp + self.fn )

        # precision
        self.pre = precision(self.tp, self.fp)

        # Recall
        self.rec = recall(self.tp, self.fn)

        #F-score
        self.fscore = fScore(self.pre, self.rec)

        # Confusion Matrix
        self.cm = [[self.tp,self.fp],[self.fn, self.tn]]
        self.cm = np.matrix(self.cm)
        
        self.scores = { 'accuracy': self.acc,'precision': self.pre, 'recall': self.rec, 'f1_score': self.fscore}

        return self.scores, self.cm