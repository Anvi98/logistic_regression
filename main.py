import pandas as pd
from module import *
import model

if __name__ == '__main__':

    # Load data sets
    X = pd.read_csv("features.csv")
    Y = pd.read_csv("labels.csv")

    Y.replace('ham', 0, inplace=True)
    Y.replace('spam', 1, inplace=True)

    X = np.array(X)
    X = X.transpose()
    Y = np.array(Y)

    ## split dataset
    # train 80/20
    X_train, Y_train, X_test, Y_test = splitter(X, Y, 0.8)

    model1 = model.LogisticRegAlex()
    W, history = model1.train(X_train, Y_train, 1000)
    scores_train, cm_train = model1.test(X_train, Y_train)
    scores, cm = model1.test(X_test, Y_test)

    # Plot Confusion Matrix with Matplotlib
    plotConfusionMatrix(cm)

    # Plot Learning Curve with matplotlib
    plotLearningCurve(history)

    # Print Metrics
    print(f'Metrics on Training set: {scores_train}')
    print(f'Metrics on Testing set: {scores}')