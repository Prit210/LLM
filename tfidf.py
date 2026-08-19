# %%
import numpy as np
import pandas as pd
from collections import Counter
import math
import glob
from tqdm import tqdm
import pickle
import os
import json


# %% [markdown]
# Build vocabulary from training data

# %%
def tokenize(sentence):
    return str(sentence).split()

#

# %%

# def tf_old(sentence,vocab,word_to_idx):
#     tf_values=np.zeros(len(vocab),dtype=np.uint16)
#     temp={}

#     tokens=tokenize(sentence)

#     for t in tokens:
#             temp[t]=temp.get(t,0)+1

#     for word,freq in temp.items():
#          if word in vocab:
#               tf_values[word_to_idx[word]]=freq

#     return tf_values

# def tf_slow(sentence, word_to_idx):

#     tf_value = np.zeros(len(word_to_idx),dtype=np.uint16)

#     for token in str(sentence).split():
#         idx = word_to_idx.get(token)

#         if idx is not None:
#             tf_value[idx] += 1

#     return tf_value

def tf(sentence, word_to_idx):

    tf_value={}

    tokens=tokenize(sentence)

    for token in tokens:
        idx = word_to_idx.get(token)

        if idx is not None:
            tf_value[idx]=tf_value.get(idx,0)+1

    return tf_value



# %% [markdown]
# Calculate IDF (Inverse Document Frequency)

# %%
# def idf_old(train_df,vocab,word_to_idx):
#     N = len(train_df)  # Total documents
#     df_value=np.zeros(len(word_to_idx),dtype=np.uint16)

#     for sentence in tqdm(train_df,total=N,desc="Computing IDF"):
#         tokens=set(tokenize(sentence))

#         for t in tokens:
#             if t in vocab:
#                 df_value[word_to_idx[t]]=df_value[word_to_idx[t]]+1

#     idf_values=np.log((N+1)/(df_value+1))+1

#     return idf_values


def idf(train_df,word_to_idx):
    N = len(train_df)  # Total documents
    df_value=np.zeros(len(word_to_idx),dtype=np.uint16)

    for sentence in tqdm(train_df,total=N,desc="Computing IDF"):
        tokens=set(tokenize(sentence))

        for t in tokens:
            idx=word_to_idx.get(t)
            if idx is not None:
                df_value[idx]=df_value[idx]+1

    idf_values=np.log((N+1)/(df_value+1))+1

    return idf_values

# %% [markdown]
# Calculate TF-IDF for each document

# %%
# def calculate_tfidf_old(sentences, idf_values, vocab, word_to_idx):
    
#     """Calculate TF-IDF vectors for sentences"""
#     tfidf_matrix = {}
    
#     for doc_idx, sentence in tqdm(enumerate(sentences), total=len(sentences), desc="Computing TF-IDF"):
        
#         # Calculate term frequency
#         tf_value = tf_slow(sentence,word_to_idx)
        
#         tfidf=tf_value*idf_values

#         result = {word: tfidf[idx] for word, idx in word_to_idx.items() if tfidf[idx] != 0 }
#         tfidf_matrix[doc_idx]=result
    
#     return tfidf_matrix

def calculate_tfidf(sentences, idf_values, word_to_idx):
    
    """Calculate TF-IDF vectors for sentences"""
    tfidf_matrix = {}
    
    for doc_idx, sentence in tqdm(enumerate(sentences), total=len(sentences), desc="Computing TF-IDF"):
        
        # Calculate term frequency
        tf_values = tf(sentence,word_to_idx)
        
        result={}
        for idx,tf_v in tf_values.items():
            temp=tf_v*idf_values[idx]

            if temp != 0:
                result[idx]=temp

        tfidf_matrix[doc_idx]=result
    
    return tfidf_matrix
