import pandas as pd 
import numpy as np
import nltk

raw = pd.read_csv('SMS_spam_corpus_big.csv', sep=',',encoding ='latin1')

# The data is unbalanced and it looks ordered so I shuffle it to help the model learns better
raw = raw.sample(frac=1, random_state=42)
raw_X = raw['messages']
raw_Y = raw["labels"]

r = list(raw_X)

# Pos_tagging
pos_sentences = [ nltk.pos_tag(nltk.word_tokenize(sentence)) for sentence in r]
for t in pos_sentences:
    tmp = []
    i = pos_sentences.index(t)
    for tuple in t:
        tmp.append(tuple[0])
    
    pos_sentences[i] = tmp

# Tokenization
sent_tokens = []
for i in range(len(pos_sentences)):
    # Here I am getting some errors because some messages are already of type list. So to bypass it I set an if condition
    if type(raw_X[i]) == list:
        next
    else:
        #Split by space => unigram tokenization in this particular case.
        sentence = raw_X[i].split()
        raw_X.values[i] = sentence

## Unique tokens
unique_tokens =  []
for list in pos_sentences:
    for token in list:
        if token not in unique_tokens:
            unique_tokens.append(token)
        else:
            next

# Frequencies of words in each text message
freqs = {}

for t in unique_tokens:
    tmp_count = []
    for m in pos_sentences:
        t_m = m.count(t)
        tmp_count.append(t_m)
    freqs[t] = tmp_count

freq_columns = {}

# That's the total number of messages 
size_value_freq = 1324
for j in range(size_value_freq):
    tmp_n = []
    for i, v in freqs.items():
        tmp_n.append(v[j])
    freq_columns[unique_tokens[j]] = tmp_n

#Normalize frequencies
n_freqs = {}
for j in range(size_value_freq):
    tmp_n = []
    for k, v in freq_columns.items():
        tmp = v[j] / len(pos_sentences[j])
        tmp_n.append(tmp)
    n_freqs[unique_tokens[j]] = tmp_n      

normalized_freq = [f for f in n_freqs.values()]

features = pd.DataFrame(normalized_freq)

features.to_csv('features.csv', index=False)
raw_Y.to_csv('labels.csv', index=False)