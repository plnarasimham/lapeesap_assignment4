#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function

import re
import sys
import numpy as np
from operator import add

from pyspark import SparkContext
from math import exp

sc = SparkContext()


def freqArray (listOfIndices, numberofwords):
    returnVal = np.zeros (20000)
    for index in listOfIndices:
        returnVal[index] = returnVal[index] + 1
    returnVal = np.divide(returnVal, numberofwords)
    return returnVal

# Reading the file into RDD
d_corpus = sc.textFile(sys.argv[1], 1)
#d_corpus = sc.textFile("SmallTrainingData.txt",10)
d_keyAndText = d_corpus.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))
regex = re.compile('[^a-zA-Z]')

d_keyAndListOfWords = d_keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))

#d_keyAndListOfWords.take(2)


d_word = d_keyAndListOfWords.flatMap(lambda x: [(word,1) for word in x[1]]).reduceByKey(add)
#d_word.take(2)

# total number of words
total_words = d_word.values().sum()
print("Total number of words are: ",total_words)

d_word_sorted = d_word.sortBy(lambda a: -a[1])
#d_word_sorted.take(2)

#d_word_sorted.top(2)

d_word_top20k = d_word_sorted.take(20000)
type(d_word_top20k)

d_word_top20k

topWordsK = sc.parallelize(range(20000))

dictionary = topWordsK.map(lambda x : (d_word_top20k[x][0], x))
#dictionary.take(2)

# The dictionary positions of the 5 words
test_words = dictionary.filter(lambda x: x[0] in ['applicant','and','attack','protein','car'])
print(test_words.collect())

allWordsWithDocID = d_keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))

#allWordsWithDocID.take(2)

# Now join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
allDictionaryWords = allWordsWithDocID.join(dictionary)
#allDictionaryWords.take(2)

# Now, we drop the actual word itself to get a set of (docID, dictionaryPos) pairs
justDocAndPos = allDictionaryWords.map(lambda x : x[1])

# Now get a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()
#allDictionaryWordsInEachDoc.take(2)

# Compute TF values
allDocsAsNumpyArrays = allDictionaryWordsInEachDoc.map(lambda x: (x[0],freqArray(x[1],20000)))
#allDocsAsNumpyArrays_3 = allDocsAsNumpyArrays.take(3)
#print(allDocsAsNumpyArrays_3)

allDocsAsNumpyArrays = allDocsAsNumpyArrays.map(lambda x: (1 if x[0][:2]=="AU" else 0,x[1]))
#allDocsAsNumpyArrays.take(3)


# Computing the coeff using gradient descent

num_iteration = 20
learningRate=0.0000001
r = np.zeros(20000)
gradientCost = 0
lam = 100
l2_norm_threshold = 0.0015
l2_norm = 100000
iter =0

while(iter<20):
    gradientCost=allDocsAsNumpyArrays.map(lambda x: -x[1]*x[0] + x[1]*(exp(np.dot(x[1],r))/(1+exp(np.dot(x[1],r))))).reduce(lambda x, y: x+y)
    gradientCost = gradientCost+2*lam*r
    print("gradient cost is:",gradientCost)
    
    r_new = r - learningRate * gradientCost
    r_diff = r-r_new
    l2_norm = np.linalg.norm(r_diff)
    print("L2-norm is:",l2_norm)
    r = r_new
    iter = iter+1


# finding the largest weights
r_zip = list(zip(list(range(20000)),r))

r_zip.sort(key=lambda y: y[1],reverse=True)

# Top Keywords to predict Australian cour cases
indices = [x[0] for x in r_zip[:5]]
keywords = dictionary.filter(lambda x: x[1] in indices)
print(keywords.collect())


# Putting the model on the test data
d_corpus_test = sc.textFile(sys.argv[2], 1)
d_keyAndText_test = d_corpus_test.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))
regex = re.compile('[^a-zA-Z]')

d_keyAndListOfWords_test = d_keyAndText_test.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))

# Joining with dictionary
d_word_test = d_keyAndListOfWords_test.flatMap(lambda x: [(word,1) for word in x[1]]).reduceByKey(add)
allWordsWithDocID_test = d_keyAndListOfWords_test.flatMap(lambda x: ((j, x[0]) for j in x[1])).distinct()
allDictionaryWords_test = allWordsWithDocID_test.join(dictionary)
# Now, we drop the actual word itself to get a set of (docID, dictionaryPos) pairs
justDocAndPos_test = allDictionaryWords_test.map(lambda x : x[1])


# Now get a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
allDictionaryWordsInEachDoc_test = justDocAndPos_test.groupByKey()

allDocsAsNumpyArrays_test = allDictionaryWordsInEachDoc_test.map(lambda x: (x[0],freqArray(x[1],20000)))
allDocsAsNumpyArrays_test.take(3)

allDocsAsNumpyArrays_test = allDocsAsNumpyArrays_test.map(lambda x: (1 if x[0][:2]=="AU" else 0,x[1]))

# compute the predicted theta = coeff dot TF
allDocsAsNumpyArrays_test_pred = allDocsAsNumpyArrays_test.map(lambda x: (x[0],np.dot(r,x[1])))

# compute the prediction
allDocsAsNumpyArrays_test_pred_cat = allDocsAsNumpyArrays_test_pred.map(lambda x: (x[0], 1 if x[1]>0 else 0))

# Compute F1-score
true_pos = allDocsAsNumpyArrays_test_pred_cat.filter(lambda x: x[0]==1 and x[1]==1).count()
print(true_pos)
true_neg = allDocsAsNumpyArrays_test_pred_cat.filter(lambda x: x[0]==0 and x[1]==0).count()
print(true_neg)
false_pos = allDocsAsNumpyArrays_test_pred_cat.filter(lambda x: x[0]==0 and x[1]==1).count()
print(false_pos)
false_neg = allDocsAsNumpyArrays_test_pred_cat.filter(lambda x: x[0]==1 and x[1]==0).count()
print(false_neg)
F1 = true_pos/(true_pos+ 0.5*(false_pos+false_neg))
print(F1)

# Find 3 false positives
allDocsAsNumpyArrays_test_pred_cat_indexed = allDocsAsNumpyArrays_test_pred_cat.zipWithIndex()
allDocsAsNumpyArrays_test_pred_cat_indexed_fp = allDocsAsNumpyArrays_test_pred_cat_indexed.filter(lambda x: x[0][0]==0 and x[0][1]==1).take(3)
print(allDocsAsNumpyArrays_test_pred_cat_indexed_fp)



