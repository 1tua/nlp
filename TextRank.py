# ReadMe ------------------------------------------------------------------------------------------------
# Description:    script to create topn sentences in text corpus using textrank
#                 works with genism==3.8.3
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import numpy as np
import pandas as pd
import nltk
import re
from nltk.tokenize import sent_tokenize
from gensim.models import Word2Vec
from scipy import spatial
import networkx as nx

def open_file(filename):
    with open(filename,encoding="utf8") as f:
        lines = f.readlines()
        Insight_text = [item.replace("\n", ".") for item in lines]
    return Insight_text

def clean_sentences(sentences):

    sentences_clean=[re.sub(r'[^\w\s]','',sentence.lower()) for sentence in sentences]
    stop_words = stopwords.words('english')
    sentence_tokens=[[words for words in sentence.split(' ') if words not in stop_words] for sentence in sentences_clean]
    return sentence_tokens

def build_word2vec(sentence_tokens,size, min_count,iter):
    w2v=Word2Vec(sentence_tokens,size=1,min_count=1,iter=1000)
    sentence_embeddings=[[w2v[word][0] for word in words] for words in sentence_tokens]
    max_len=max([len(tokens) for tokens in sentence_tokens])
    sentence_embeddings=[np.pad(embedding,(0,max_len-len(embedding)),'constant') for embedding in sentence_embeddings]
    return sentence_embeddings

def sim_mat(sentence_tokens):
    similarity_matrix = np.zeros([len(sentence_tokens), len(sentence_tokens)])
    for i,row_embedding in enumerate(sentence_embeddings):
        for j,column_embedding in enumerate(sentence_embeddings):
            similarity_matrix[i][j]=1-spatial.distance.cosine(row_embedding,column_embedding)
    return similarity_matrix



def top_sentences(sentences,topn):
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    top_sentence={sentence:scores[index] for index,sentence in enumerate(sentences)}
    top=dict(sorted(top_sentence.items(), key=lambda x: x[1], reverse=True)[:topn])
    for sent in sentences:
        if sent in top.keys():
            return print(sent)

if __name__ == "__main__":
    #read text file
    insight_text=open_file('Risk_Insight.txt')

    #tokenize sentences
    sentences=sent_tokenize(str(insight_text))

    #preprocess sentences
    sentence_tokens=clean_sentences(sentences)

    #create sentence embeddings
    sentence_embeddings= build_word2vec(sentence_tokens,size=1,min_count=1,iter=1000)

    #finds cosine similarity 
    similarity_matrix = sim_mat(sentence_tokens)

    #Summarise speech text to 10 sentences
    top_sentences(sentences,10)
