import numpy as np
from tqdm import tqdm

def tokenize(sentence):
    return str(sentence).split()

def tf(sentence, word_to_idx):

    tf_value={}

    tokens=tokenize(sentence)

    for token in tokens:
        idx = word_to_idx.get(token)

        if idx is not None:
            tf_value[idx]=tf_value.get(idx,0)+1

    return tf_value

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
