# ReadMe ------------------------------------------------------------------------------------------------
# Description:    script to create doc with sentences that match keywords

import pandas as pd
import numpy as np
import gzip
import gensim 
import logging
import os
from gensim.models import Word2Vec
import spacy
import re
from gensim.models import KeyedVectors
import time
import pickle
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
# load model package "en_core_web_sm"
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2797059


def read_file(filename):
    
    doc = open(filename,encoding="utf8")
    return doc

def doc_without_line_brakes(filename):
    a_file = open(filename, encoding="utf8")

    string_without_line_breaks = " "
    for line in a_file:
        stripped_line = line.replace('\n', ' ')
        string_without_line_breaks += stripped_line
    a_file.close()
    return string_without_line_breaks

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


def remove_stopwords(texts):
    return [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Time calculation
def cal_elapsed_time(s):
    return print("Elapsed time:\t", round((time.time() - s),2))
s_time = time.time()
cal_elapsed_time(s=s_time)

def create_word2vec(data):
    s_time = time.time()
    print("Model Training Started...")
    w2v_model = Word2Vec(min_count=5,
                                    window=100,
                                    vector_size=150,
                                    sample=0,
                                    workers=4,
                                    batch_words=100)

    w2v_model.build_vocab(data_lemmatized)
    w2v_model.train(data_lemmatized, total_examples=w2v_model.corpus_count, total_words=w2v_model.corpus_total_words, epochs=100, compute_loss=True)

    #cal_elapsed_time(s_time)
    return w2v_model
# Save and load word2vec model
#w2v_model.save("Speech2vec.w2v_model")

#Find bigrams that contains word 
def searchlist(word):
    letters = word
    bitxt=[]
    for word in bigramlist:
        if letters in word:
            bitxt.append(word)
    bitxt=set(bitxt)
    return bitxt

#read file
doc= read_file('combined.txt')

#Convert speech to list and remove punctuation
data_corpus = list(sent_to_words(doc))

#Create bigram and trigram model
bigram = gensim.models.Phrases(data_corpus, min_count=3, threshold=5) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_corpus], threshold=3)
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
bigram_list=bigram_mod[data_corpus]
trigram_list=trigram_mod[bigram_mod[data_corpus]]

# Remove Stop Words
data_words_nostops = remove_stopwords(data_corpus)
# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)
# Form Trigrams
data_words_trigrams = make_trigrams(data_words_nostops)

# Do lemmatization keeping only noun, adj, vb, adv
#data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])`
#Load data lemmatized pickle
with open('data_lemmatized.pickle', 'rb') as f:
    data_lemmatized = pickle.load(f)

#Find bigrams with "_" in data_lemmatized
bigramlist=re.findall(r"\b\w+_\w+\b",str(data_lemmatized))

#search for risk in bigram list
bitxt =searchlist('risk')

#Remove "_" from bigrams
bitxt = [item.replace("_", " ") for item in bitxt]

#read file without line breaks
speech_doc=doc_without_line_brakes('combined.txt')

#load word2vec model
speech2vec = KeyedVectors.load("Speech2vec.w2v_model", mmap='r')

#top10 most similar words for risk
risky_list = speech2vec.wv.most_similar(positive="risk", topn=10)

#create list from first element in tuple
def create_w2vlist(list):
    w2vlist=[]
    for a in list:
            w2vlist.append(a[0])
    return w2vlist

w2vlist=create_w2vlist(risky_list)

#Find risk bigrams in speach corpus and retrieve sentence they appear in 
def retrieve_sentences(speech_doc, wordlist):
    sentences = [sentence for sentence in speech_doc.split(".") 
                if any(w.lower() in sentence.lower() for w in wordlist)]
    return sentences

bigram_sentences =retrieve_sentences(speech_doc,bitxt)
w2c_sentences =retrieve_sentences(speech_doc,w2vlist)

if __name__ == "__main__":
    #Create txt file
    with open('Risk1000_Insight.txt', 'w',encoding='utf-8') as f:
        for line in bigram_sentences:
            f.write(line)
            f.write('\n')