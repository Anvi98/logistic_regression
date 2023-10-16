import pandas as pd
from module import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, precision_score

# Load data sets

X = pd.read_csv("features.csv")
Y = pd.read_csv("labels.csv")

Y.replace('ham', 0, inplace=True)
Y.replace('spam', 1, inplace=True)

## split dataset
# train 80/20

X = np.array(X)
X = X.transpose()
Y = np.array(Y)

iteration = 1000
alpha = 0.5
cost = 0

# Save iterations and cost for plotting
history = {}

X_train, Y_train, X_test, Y_test = splitter(X, Y, 0.8)

# Initial Forward 
W = np.zeros(X_train.shape[1])
W = W.reshape(1,-1)

## Training phase
for i in range(iteration+1):
    Z = np.dot(W, X_train.T)
    y_hat = sigmoid(Z)
    y_hat = y_hat.reshape(-1,1)

    ### compute error
    # We multiplied by 0.5 to make sure the model doesn't overfit
    cost = 0.5 * np.absolute(np.mean((np.dot(-Y_train, np.log(y_hat.T)) +  np.dot((1-Y_train), np.log(1-y_hat.T)))))

    #Print cost at i iterations:
    if i % 100 == 0 or i == 1000:
        print(f'Cost: {cost}  at iteration {i}')
        # Save iterations and cost for plotting
        history[i] = cost

    ## Compute gradient
    dJ_dW = 1/len(X_train)*np.dot((y_hat - Y_train).T, X_train)

    ## Update weight with gradient descent
    W = W - alpha*dJ_dW

### Predictions 
z_test = np.dot(W, X_test.T)
y_hat_test = sigmoid(z_test)

# Threshold
y_hat_test = [1 if p > 0.5 else 0 for i in y_hat_test for p in i ]

# This will help us get a list of labels not a list of lists
Y_test = [y for i in Y_test for y in i]

# Confusion Matrix
tp, fp, tn, fn = get_confusion_matrix(Y_test, y_hat_test)

## Accuracy
acc = (tp + tn)/ (tp+ tn + fp + fn )
print(f'The accuracy on Test set is: {acc}')

y_hat = [1 if p > 0.5 else 0 for i in y_hat for p in i ]
tp_t, fp_t, tn_t, fn_t = get_confusion_matrix(Y_train, y_hat)
acc_train = (tp_t + tn_t) / (tp_t + tn_t + fp_t + fn_t)
print(f'The accuracy on train dataset is: {acc_train}')

#Create a matrix with the confusion values 
cm = [[tp,fp],[fn,tn]]
cm = np.matrix(cm)

# precision
pre = precision(tp, fp)
sk_pre = precision_score(Y_test, y_hat_test)
print(f'Precision:{pre}\n Precision with sklearn: {sk_pre}')

# Recall
rec = recall(tp, fn)
sk_rec = recall_score(Y_test, y_hat_test)
print(f'Recall:{rec} \n Recall with sklearn: {sk_rec}')

#F-score
fscore = fScore(pre, rec)
sk_f1 = f1_score(Y_test, y_hat_test)
print(f'F-score: {fscore}\n F1 with sklearn: {sk_f1}')

## Plotting
plotConfusionMatrix(cm)
plotLearningCurve(history)
