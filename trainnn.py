import numpy as np
wordsList = np.load('wordsList.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist() #Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load('wordVectors.npy')
print ('Loaded the word vectors!')

print(len(wordsList))
print(wordVectors.shape)

baseballIndex = wordsList.index('baseball')
wordVectors[baseballIndex]


import tensorflow as tf
maxSeqLength = 10 #Maximum length of sentence
numDimensions = 300 #Dimensions for each word vector
firstSentence = np.zeros((maxSeqLength), dtype='int32')
firstSentence[0] = wordsList.index("i")
firstSentence[1] = wordsList.index("thought")
firstSentence[2] = wordsList.index("the")
firstSentence[3] = wordsList.index("movie")
firstSentence[4] = wordsList.index("was")
firstSentence[5] = wordsList.index("incredible")
firstSentence[6] = wordsList.index("and")
firstSentence[7] = wordsList.index("inspiring")
#firstSentence[8] and firstSentence[9] are going to be 0
print(firstSentence.shape)
print(firstSentence) #Shows the row index for each word


with tf.Session() as sess:
    print(tf.nn.embedding_lookup(wordVectors,firstSentence).eval().shape)


from os import listdir
from os.path import isfile, join
import io
import shutil
#positiveFiles = ['positiveReviews/' + f for f in listdir('positiveReviews/') if isfile(join('positiveReviews/', f))]
positiveFiles = ['12_5k_SeriousArticles/' + f for f in listdir('12_5k_SeriousArticles/') if isfile(join('12_5k_SeriousArticles/', f))]
#negativeFiles = ['negativeReviews/' + f for f in listdir('negativeReviews/') if isfile(join('negativeReviews/', f))]

negativeFiles = ['12_5k_SatireArticles/' + f for f in listdir('12_5k_SatireArticles/') if isfile(join('12_5k_SatireArticles/', f))]
numWords = []
maxLen = -1
for pf in positiveFiles:
    with io.open(pf, "r", encoding='utf-8', errors='ignore') as f:
        line=f.readline()
	#if(maxLen == -1):
	#	print(line)
	#	print('\n')
#	if(len(line) > maxLen):
#	        maxLen = len(line)
	counter = len(line.split())
        numWords.append(counter)       
print('Serious files finished')
print('max is : ', maxLen)
for nf in negativeFiles:
    with io.open(nf, "r", encoding='utf-8', errors='ignore') as f:
        line=f.readline()
#	if(len(line) > maxLen):
#	        maxLen = len(line)
        counter = len(line.split())
        numWords.append(counter)  
print('Satire files finished')
print('max is : ', maxLen)
numFiles = len(numWords)
print('The total number of files is', numFiles)
print('The total number of words in the files is', sum(numWords))
print('The average number of words in the files is', sum(numWords)/len(numWords))

#print('dddlltta')
#import matplotlib.pyplot as plt
#%matplotlib inline
#plt.hist(numWords, 50)
#plt.xlabel('Sequence Length')
#plt.ylabel('Frequency')
#plt.axis([0, 1200, 0, 8000])
#plt.show()


maxSeqLength = 400

# Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
import re
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

fname = positiveFiles[3] #Can use any valid index (not just 3)
with open(fname) as f:
    print(fname)
    for lines in f:
        print(lines)
        exit

firstFile = np.zeros((maxSeqLength), dtype='int32')
with open(fname) as f:
    indexCounter = 0
    line=f.readline()
#    print('****************')
#    print(line)
#    print('****************')
    cleanedLine = cleanSentences(line)
    split = cleanedLine.split()
    for word in split:
        if indexCounter < maxSeqLength:
            try:
                firstFile[indexCounter] = wordsList.index(word)
            except ValueError:
                firstFile[indexCounter] = 399999 #Vector for unknown words
        indexCounter = indexCounter + 1
firstFile

#For making ids matrix 
# print('making id matrix')

# ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
# fileCounter = 0
# for pf in positiveFiles:
   # with open(pf, "r") as f:
       # indexCounter = 0
       # line=f.readline()
       # cleanedLine = cleanSentences(line)
       # split = cleanedLine.split()
       # for word in split:
           # try:
               # ids[fileCounter][indexCounter] = wordsList.index(word)
           # except ValueError:
               # ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
           # indexCounter = indexCounter + 1
           # if indexCounter >= maxSeqLength:
                # print("break")
                # break
       # fileCounter = fileCounter + 1 
# print('positive matrix done')
# for nf in negativeFiles:
   # with open(nf, "r") as f:
       # indexCounter = 0
       # line=f.readline()
       # cleanedLine = cleanSentences(line)
       # split = cleanedLine.split()
       # for word in split:
           # try:
               # ids[fileCounter][indexCounter] = wordsList.index(word)
           # except ValueError:
               # ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
           # indexCounter = indexCounter + 1
           # if indexCounter >= maxSeqLength:
                # print("break")
                # break
       # fileCounter = fileCounter + 1 
 # #Pass into embedding function and see if it evaluates. 

# np.save('idsMatrix', ids)


# print('id matrix done\n')

ids = np.load('idsMatrix.npy')


from random import randint
print("len ++++ ", len(ids))
def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0): 
            num = randint(1,11499)
            labels.append([1,0])
        else:
            num = randint(13499,24999)
            labels.append([0,1])
	#print('done')
	#print(ids[0:20])
	print("num : ", num)	
	#print(ids[num-10:num+10])
        arr[i] = ids[num-1:num]
    return arr, labels

def getTestBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(11499,13499)
        if (num <= 12499):
            labels.append([1,0])
        else:
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels


#RNN

batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 100000

import tensorflow as tf
tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

import datetime

tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

#For training the model

# sess = tf.InteractiveSession()
# saver = tf.train.Saver()
# sess.run(tf.global_variables_initializer())

# for i in range(iterations):
  # #  Next Batch of reviews
  # nextBatch, nextBatchLabels = getTrainBatch();
  # sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})
   
   # #Write summary to Tensorboard
  # if (i % 50 == 0):
      # summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
      # writer.add_summary(summary, i)

  # #  Save the network every 10,000 training iterations
  # if (i % 10000 == 0 and i != 0):
      # save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
      # print("saved to %s" % save_path)
# writer.close()

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('models'))

iterations = 10
for i in range(iterations):
    nextBatch, nextBatchLabels = getTestBatch();
    print("Accuracy for this batch:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)

import re
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

def getSentenceMatrix(sentence):
    arr = np.zeros([batchSize, maxSeqLength])
    sentenceMatrix = np.zeros([batchSize,maxSeqLength], dtype='int32')
    cleanedSentence = cleanSentences(sentence)
    split = cleanedSentence.split()
    for indexCounter,word in enumerate(split):
        try:
            sentenceMatrix[0,indexCounter] = wordsList.index(word)
        except ValueError:
            sentenceMatrix[0,indexCounter] = 399999 #Vector for unkown words
    return sentenceMatrix

