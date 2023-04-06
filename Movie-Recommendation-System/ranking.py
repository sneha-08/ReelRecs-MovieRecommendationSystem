import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import bs4 as bs
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances


def create_similarity():
    data = pd.read_csv('main_data.csv')
    # creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # creating a similarity score matrix
    similarity = cosine_similarity(count_matrix)
    return data,similarity


def rank_tf_idf():
    data = pd.read_csv('main_data.csv')
    # Create the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer to the documents and transform the documents into vectors
    tf_idf_vectors = vectorizer.fit_transform(data)

    # Get the feature names (terms) from the vectorizer
    terms = vectorizer.get_feature_names()

    # # Print the TF-IDF vectors for each document
    # for i, data in enumerate(data):
    #     #print("TF-IDF vector for document", i+1)
    #     for j, term in enumerate(terms):
    #         print(term, ":", tf_idf_vectors[i, j])
    #     print()
    return vectorizer


def rank_bm25():
    data = pd.read_csv('main_data.csv')

    # Preprocess the documents by splitting them into tokens
    tokenized_documents = [document.split() for document in data]

    # Create the BM25Okapi object
    bm25 = BM25Okapi(tokenized_documents)

    # Create a query
    query = ""

    # Get the document scores for the query
    scores = bm25.get_scores(query.split())

    # Print the scores for each document
    for i, score in enumerate(scores):
        print("BM25 score for document", i+1, ":", score)
    return score

def jaccard_similarity():
    data = pd.read_csv('main_data.csv')
    # creating a count matrix
    cv = CountVectorizer(tokenizer=lambda doc: doc.lower().split())
    count_matrix = cv.fit_transform(data['comb'])
    # creating a similarity score matrix
    jsimilarity = 1 - pairwise_distances(count_matrix, metric='jaccard')
    return jsimilarity

def peasrson_correlation():

    # read in data as a pandas DataFrame
    data = pd.read_csv('data.csv')

    # select two columns to calculate correlation
    x = data['data[3]']
    y = data['genre']

    # calculate mean of each column
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # calculate standard deviation of each column
    x_std = np.std(x)
    y_std = np.std(y)

    # calculate covariance between the two columns
    cov = np.sum((x - x_mean) * (y - y_mean)) / len(x)

    # calculate Pearson correlation coefficient
    corr = cov / (x_std * y_std)
    return corr