# Author: Anvi Alex Eponon
# Date : 16/10/2023
# Type : Preprocessing Task.
# Description: Here I just did very few preprocessing to be able to work with the file in a csv format
#--------------------

from module import *
import nltk
import spacy
import pandas as pd

## --------- Prepare the document for data preprocessing----------
# load only the english language
nlp = spacy.load("en_core_web_sm")

# have to specify the encoding as latin1 to be able to open the file and keep special characters
with open("SMS_Spam_Corpus_big.txt", "r", encoding= "windows-1252") as file:
    data = file.readlines()

# Another observation will be that putting sentence tokens in their lema forms doesn't affect too much the inference
sentences_pool = [ lemmatize_text(sentence, nlp) for sentence in data]

# Tokenization
X = []
Y = []
word_tokenizer = nltk.word_tokenize

for sentence in sentences_pool:
    sentence = word_tokenizer(sentence)
    x, y = txtSplitter(sentence)
    X.append(x)
    Y.append(y)

# lower all tokens
X = [[token.lower() for token in sentence] for sentence in X]

#Remove extra commas in sentences
commaRemover(X)

#Regroup words into sentences
sentenceTokenizer(X)

# Create Dataframe
X = pd.DataFrame(X, columns=['messages'])
Y = pd.DataFrame(Y, columns=['labels'])

df = pd.concat([X, Y], axis=1)

# Save data to csv form
df.to_csv("SMS_spam_corpus_big.csv", encoding='windows-1252', index=False)
