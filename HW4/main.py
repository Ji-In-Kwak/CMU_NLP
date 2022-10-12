#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Abhishek Srivastava <asrivas4@andrew.cmu.edu>

"""
11-411/611 NLP Assignment 3
This file imports your implemented class/methods and runs it using the provided arguments. Don't edit this file.

Example command:
python3 main.py --train data/lyrics/taylor_swift.txt --test data/lyrics/test_lyrics.txt --n 3 --smoothing 0 --min_freq 1
"""

from utils import *
from lm import *
        
def main(args):
    #-------------- Reading Data --------------#
    train = read_file(args.train)
    test = read_file(args.test)

    print("--------------------------------------------------")
    print("No of sentences in train file: {}".format(len(train)))
    print("--------------------------------------------------")
    print("No of sentences in test file: {}".format(len(test)))
    print("--------------------------------------------------")
    print("Raw train example: {}".format(train[2]))
    print("--------------------------------------------------")
    print("Raw test example: {}".format(test[2]))
    print("--------------------------------------------------")

    #-------------- Preprocessing --------------#
    train = preprocess(train, args.n)
    test = preprocess(test, args.n)

    print("Preprocessed train example: \n{}\n".format(train[2]))
    print("--------------------------------------------------")
    print("Preprocessed test example: \n{}".format(test[2]))
    print("--------------------------------------------------")

    # Change train tokens to UNK whenever freq < min_freq
    train = change_to_unk(train, args.min_freq)
    # Change test tokens to UNK whenever test word OOV (not present in train_vocab)
    train_vocab = Counter(flatten(train))
    test = change_to_unk(test, args.min_freq, train_vocab)

    #-------------- Language Model --------------#
    print("Loading {}-gram model.".format(args.n))
    lm = LanguageModel(args.n, train, args.smoothing)

    print("Vocabulary size (unique unigrams): {}".format(len(lm.vocab)))
    print("Total number of unique n-grams: {}".format(len(lm.model)))
    print("--------------------------------------------------")

    #-------------- Perplexity --------------#
    ppl = perplexity(lm=lm, test_data=test)
    print("Model perplexity: {:.3f}".format(ppl))
    print("--------------------------------------------------")

    #-------------- Sentence Completion --------------#
    print("Generating random sentences.")
    num_to_generate = 5
    for sentence, prob in generate_sentences(lm, num_to_generate):
        print("{} ({:.5f})".format(sentence, prob))
        
    print("--------------------------------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Your n-gram language model.")
    parser.add_argument('--train', type=str, required=True,
            help='Location of train file (.txt format)')
    parser.add_argument('--test', type=str, required=True,
            help='Location of test file (.txt format)')
    parser.add_argument('--n', type=int, required=True,
            help='Order of n-gram model to create (i.e. 1 for unigram, 2 for bigram, etc.)')
    parser.add_argument('--smoothing', type=int, default=1,
            help='Flag for performing laplace smoothing or not.')
    parser.add_argument('--min_freq', type=int, default=1,
            help='Min freq threshold for every word. Replace with <UNK> if freq < min_freq.')
    args = parser.parse_args()
    main(args)