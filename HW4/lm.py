#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Ishita <igoyal@andrew.cmu.edu> and Suyash <schavan@andrew.cmu.edu> based on work by Abhishek <asrivas4@andrew.cmu.edu>

"""
11-411/611 NLP Assignment 4
N-gram Language Model Implementation

Complete the LanguageModel class and other TO-DO methods.
"""

#######################################
# Import Statements
#######################################
from utils import *
from collections import Counter
from itertools import product
import argparse
import random
import math
from collections import defaultdict

#######################################
# Helper Functions
#######################################
def flatten(lst):
    """
    Flattens a nested list into a 1D list.
    Args:
        lst: Nested list (2D)
    
    Returns:
        Flattened 1-D list
    """
    return [item for sublist in lst for item in sublist]


#######################################
# TO-DO: get_ngrams()
#######################################
def get_ngrams(list_of_words, n):
    """
    Returns a list of n-grams for a list of words.
    Args
    ----
    list_of_words: List[str]
        List of already preprocessed and flattened (1D) list of tokens e.g. ["<s>", "hello", "</s>", "<s>", "bye", "</s>"]
    n: int
        n-gram order e.g. 1, 2, 3
    
    Returns:
        n_grams: List[Tuple]
            Returns a list containing n-gram tuples
    """
#     return NotImplemented
    n_grams = []
    for i in range(len(list_of_words) - n + 1):
        n_grams.append(tuple(list_of_words[i:i+n]))
        
    
    return n_grams

#######################################
# TO-DO: LanguageModel()
#######################################
class LanguageModel():
    def __init__(self, n, train_data, alpha=1):
        """
        Language model class.
        
        Args
        ____
        n: int
            n-gram order
        train_data: List[List]
            already preprocessed list of sentences. e.g. [["<s>", "hello", "my", "</s>"], ["<s>", "hi", "there", "</s>"]]
        alpha: float
            Smoothing parameter
        
        Other required parameters:
            self.vocab: vocabulary dict with counts
            self.model: n-gram language model, i.e., n-gram dict with probabilties
            self.n_grams_counts: Count of all the n-grams present in the training data
            self.prefix_counts: Count of all the corresponding n-1 grams present in the training data
        """
        self.n = n
        self.train_data = train_data
        self.tokens = flatten(self.train_data)
        
        self.n_grams_counts = None
        self.prefix_counts = None
        self.vocab  = Counter(self.tokens)
        self.alpha = alpha
        self.model = self.build()
        
    def get_smooth_probabilites(self,n_gram):
        """
        Returns the smoothed probability of the ngram, using Laplace Smoothing
        """
        
        if self.n == 1:
            val = (self.n_grams_counts[n_gram] + self.alpha) / (len(self.tokens) + self.alpha*len(self.vocab))
        else:
            val = (self.n_grams_counts[n_gram] + self.alpha) / (self.prefix_counts[n_gram[:-1]] + self.alpha*len(self.vocab))

        return val
    

    
    def build(self):
        """
        Returns a n-gram (could be a unigram) dict with probabilities
        """
        # TODO: Get the n-grams from the training data using the previously defined methods
        self.n_grams = get_ngrams(self.tokens, self.n)
        
        # TODO: Define the class variables n_grams_counts and prefix_counts 
        self.n_grams_counts = Counter(self.n_grams)
#         if self.n > 1:
        self.prefix_counts = Counter(get_ngrams(self.tokens, self.n-1))

        # TODO Get the Probabilities using the get_smooth_probabilities
#         prob = self.get_smooth_probabilites(self.n_grams)
        prob = dict()
        for n_gram in self.n_grams:
            prob[n_gram] = self.get_smooth_probabilites(n_gram)

        return prob


#######################################
# TO-DO: perplexity()
#######################################
def perplexity(lm, test_data):
    """
    Returns perplexity calculated on the test data.
    Args
    ----------
    test_data: List[List] 
        Already preprocessed nested list of sentences
        
    lm: LanguageModel class object
        To be used for retrieving lm.model, lm.n and lm.vocab

    Returns
    -------
    float
        Calculated perplexity value
    """
    # TODO Flatten and get the n-grams
    tokens = flatten(test_data)
    n_grams = get_ngrams(tokens, n = lm.n)
#     print("n_grams \t :", n_grams)
    
    N = len(n_grams)
#     if lm.n == 1:
    perplexity = math.exp(sum([math.log((1/lm.get_smooth_probabilites(w)) ** (1/N)) for w in n_grams]))
#     else:
#         perplexity = math.exp(sum([math.log((1/lm.model[n_gram])**(1/N)) for n_gram in n_grams]))
    # TODO Calculate the Perplexity over all the test n-grams
#     return NotImplemented
    return perplexity


###############################################
# Method: Most Probable Candidates [Don't Edit]
###############################################
def best_candidate(lm, prev, i, without=[], mode="random"):
    """
    Returns the most probable word candidate after a given sentence.
    """
    blacklist  = ["<UNK>"] + without
    candidates = ((ngram[-1],prob) for ngram,prob in lm.model.items() if ngram[:-1]==prev)
    candidates = filter(lambda candidate: candidate[0] not in blacklist, candidates)
    candidates = sorted(candidates, key=lambda candidate: candidate[1], reverse=True)
    if len(candidates) == 0:
        return ("</s>", 1)
    else:
        if(mode=="random"):
            return candidates[random.randrange(len(candidates))]
        else:
            return candidates[0]

def top_k_best_candidates(lm, prev, k, without=[]):
    """
    Returns the K most-probable word candidate after a given n-1 gram.
    Args
    ----
    lm: LanguageModel class object
    
    prev: n-1 gram
        List of tokens n
    """
    blacklist  = ["<UNK>"] + without
    candidates = ((ngram[-1],prob) for ngram,prob in lm.model.items() if ngram[:-1]==prev)
    candidates = filter(lambda candidate: candidate[0] not in blacklist, candidates)
    candidates = sorted(candidates, key=lambda candidate: candidate[1], reverse=True)
    if len(candidates) == 0:
        return ("</s>", 1)
    else:
        return candidates[:k]
        
###########################################
# Method: Generate Sentences [Don't Edit]
###########################################
def generate_sentences_from_phrase(lm, num, sent, prob, mode):
    """
    Generate sentences using the trained language model after a
    provided phrase.
    """
    min_len=12
    max_len=24

    sentences = []
    for i in range(num):
        while sent[-1] != "</s>":
            prev = () if lm.n == 1 else tuple(sent[-(lm.n-1):])
            blacklist = sent + (["</s>"] if len(sent) < min_len else [])

            next_token, next_prob = best_candidate(lm, prev, i, without=blacklist, mode=mode)
            sent.append(next_token)
            prob *= next_prob
            
            if len(sent) >= max_len:
                sent.append("</s>")

        sentences.append((' '.join(sent), -1/math.log(prob)))

    return sentences

def generate_sentences(lm, num, mode="random"):
    """
    Generate sentences using the trained language model without any 
    provided phrase to begin with.
    """
    min_len=12
    max_len=24

    sentences = []
    for i in range(num):
        sent, prob = ["<s>"] * max(1, lm.n-1), 1
        while sent[-1] != "</s>":
            prev = () if lm.n == 1 else tuple(sent[-(lm.n-1):])
            blacklist = sent + (["</s>"] if len(sent) < min_len else [])

            next_token, next_prob = best_candidate(lm, prev, i, without=blacklist, mode=mode)
            sent.append(next_token)
            prob *= next_prob
            
            if len(sent) >= max_len:
                sent.append("</s>")

        sentences.append((' '.join(sent), -1/math.log(prob)))

    return sentences

# Copy of main executable script provided locally for your convenience
# This is not executed on autograder, so do what you want with it
if __name__ == '__main__':
    train = "data/sample.txt"
    test = "data/sample.txt"
    n = 2
    alpha = 0

    print("No of sentences in train file: {}".format(len(train)))
    print("No of sentences in test file: {}".format(len(test)))

    print("Raw train example: {}".format(train[2]))
    print("Raw test example: {}".format(test[2]))

    train = preprocess(train, n)
    test = preprocess(test, n)

    print("Preprocessed train example: \n{}\n".format(train[2]))
    print("Preprocessed test example: \n{}".format(test[2]))

    # Language Model
    print("Loading {}-gram model.".format(n))
    lm = LanguageModel(n, train, alpha)

    print("Vocabulary size (unique unigrams): {}".format(len(lm.vocab)))
    print("Total number of unique n-grams: {}".format(len(lm.model)))
    
    # Perplexity
    ppl = perplexity(lm=lm, test_data=test)
    print("Model perplexity: {:.3f}".format(ppl))
    
    # Generating sentences using your model
    print("Generating random sentences.")
    num_to_generate = 5
    for sentence, prob in generate_sentences(lm, num_to_generate):
        print("{} ({:.5f})".format(sentence, prob))