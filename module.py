import numpy as np 
import matplotlib.pyplot as plt

def splitter(X, Y, percent):
    X_train = X[:int(len(X)*percent)]
    Y_train = Y[:int(len(Y)*percent)]
    
    X_test = X[int(len(X)*percent):] 
    Y_test = Y[int(len(Y)*percent):]

    return X_train, Y_train, X_test, Y_test

## Confusion matrix
def get_confusion_matrix(Y, y_hat):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    # True Positive and False positive
    for l, p in zip(Y, y_hat):
        if l == 1 and p ==1:
            tp +=1
        if l == 0 and p == 1:
            fp +=1
        if l == 1 and p == 0:
            fn +=1
        if l == 0 and p == 0:
            tn +=1
    return tp, fp, tn, fn

## Precision
def precision(tp, fp):
    precision = tp / (tp + fp)
    return precision

## Recall
def recall(tp, fn):
    recall = tp / (tp + fn)
    return recall

## F-measure
def fScore(precision, recall):
    fscore = (2*precision*recall)/(precision + recall)
    return fscore

# sigmoid
def sigmoid(z):
    return 1 / (1+ np.exp(-(z)))

def plotConfusionMatrix(cm):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()

def plotLearningCurve(history):
    x = list(history.keys())
    y = list(history.values())
    plt.plot(x, y)
    plt.title("Curve plotted using the given x and y")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def txtSplitter(text):
    x = text[:-1]
    y = text[-1]
    return x, y

def lemmatize_text(text, nlp):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

def commaRemover(tokens_list):
    for tokens in tokens_list:
        for token in tokens:
            if token == ',':
                tmp_index = tokens.index(token)
                del tokens[tmp_index]
    #return tokens_list

def sentenceTokenizer(tokens_list):
    for tokens in tokens_list:
        temp_sent = ' '.join(tokens)
        tmp_index = tokens_list.index(tokens)
        tokens_list[tmp_index] = temp_sent