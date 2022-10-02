
# author:bears
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import wordnet as wn
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# read the data
df = pd.read_csv("data/labelled_movie_reviews.csv")

# shuffle the rows
df = df.sample(frac=1, random_state=123).reset_index(drop=True)

# get the train, val, test splits
train_frac, val_frac, test_frac = 0.7, 0.1, 0.2
Xr = df["text"].tolist()
Yr = df["label"].tolist()
train_end = int(train_frac * len(Xr))
val_end = int((train_frac + val_frac) * len(Xr))
X_train = Xr[0:train_end]
Y_train = Yr[0:train_end]
X_val = Xr[train_end:val_end]
Y_val = Yr[train_end:val_end]
X_test = Xr[val_end:]
Y_test = Yr[val_end:]

data = dict(np.load("data/word_vectors.npz"))
w2v = {w: v for w, v in zip(data["words"], data["vectors"])}

# initialize a tokenizer
tokenizer = TreebankWordTokenizer()



# convert a document into a vector
def document_to_vector(corpus):
    """Takes corpus and turns it into a vector array
    by aggregating its word temp.
    Here Each review is tokenized first and checked in the w2v , and vector is added.
    for missing words synonyms are found using wordnet.
    Again each word in synonym list is checked in w2v and first occurence of synonym's vector is added.
    Args:
        corpus (list) : The corpus consists of all reviews
    Returns:
        np.array: The word vector this will be 300 dimension.
    """
    vec = []
    for review in corpus:
        doc_tok = tokenizer.tokenize(review)
        filtered_words = []
        missing_words = []
        for i in doc_tok:
            if i not in w2v:
                lemma_all = []
                for synset in wn.synsets(i[:-1]):
                    for lemma in synset.lemmas():
                        lemma_all.append(lemma.name())
                for j in lemma_all:
                    if j in w2v:
                        missing_words.append(j)
                        break
            else:
                filtered_words.append(w2v[i])
        for i in missing_words:
            filtered_words.append(w2v[i])
        vec.append(np.mean(np.array(filtered_words), axis=0))
    return vec


# fit a linear model
def fit_model(Xtr, Ytr, C):
    """Given a training dataset and a regularization parameter
        return a linear model fit to this data.
    Args:
        Xtr (list(str)): The input training examples. Each example is a
            document as a string.
        Ytr (list(str)): The list of class labels, each element of the
            list is either 'neg' or 'pos'.
        C (float): Regularization parameter C for LogisticRegression
    Returns:
        LogisticRegression: The trained logistic regression model.
    """
    
    Xtr_vec = document_to_vector(Xtr)
    lr = LogisticRegression(C=C)
    model = lr.fit(Xtr_vec, Ytr)
    return model


# fit a linear model
def test_model(model, Xtst, Ytst):
    """Given a model already fit to the data return the accuracy
        on the provided dataset.
    Args:
        model (LogisticRegression): The previously trained model.
        Xtst (list(str)): The input examples. Each example
            is a document as a string.
        Ytst (list(str)): The input class labels, each element
            of the list is either 'neg' or 'pos'.
    Returns:
        float: The accuracy of the model on the data.
    """
    
    Xtst_vec = document_to_vector(Xtst)
    score = model.score(Xtst_vec, Ytst)
    return score


C_set = [0.01,0.05, 0.1, 0.4,0.8,1, 5,7,10,20, 25]
score_list=[]
for i in C_set:
      model= fit_model(np.array(X_train), Y_train,i)
      score = test_model(model, np.array(X_val), Y_val)
      score_list.append(score)
      print('C:',i,'score:',score)

C_best_ind=score_list.index(max(score_list))
C_best = C_set[C_best_ind]
print('C_best:',C_best)


X_tr = Xr[0:val_end]
Y_tr = Yr[0:val_end]
X_ts = Xr[val_end:]
Y_ts = Yr[val_end:]
best_model=fit_model(X_tr,Y_tr,C_best)
best_acc = test_model(best_model,X_ts,Y_ts)
print("Best Test Accuracy :",best_acc)