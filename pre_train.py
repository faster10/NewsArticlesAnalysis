numDimensions = 300
maxSeqLength = 400
batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 100000

import numpy as np
wordsList = np.load('wordsList.npy').tolist()
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load('wordVectors.npy')

import tensorflow as tf
tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.25)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('models'))

# Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
import re
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

def getSentenceMatrix(sentence):
    arr = np.zeros([batchSize, maxSeqLength])
    sentenceMatrix = np.zeros([batchSize,maxSeqLength], dtype='int32')
    #cleanedSentence = cleanSentences(sentence)
    split = sentence.split()
    for indexCounter,word in enumerate(split):
        try:
            sentenceMatrix[0,indexCounter] = wordsList.index(word)
        except ValueError:
            sentenceMatrix[0,indexCounter] = 399999 #Vector for unkown words
    return sentenceMatrix
print(chr(27) + "[2J")
print("Test Articles....")
print("\n")


inputText = "\"We really need camps for adults,\" Hillary Clinton said, citing a \"fun deficit\" in America. \"You can have the red cabin, the blue cabin, have to come together and actually listen to each other.\" It was a light moment during the presidential hopeful's appearance here Thursday, tailored to her audience: hundreds of camp professionals gathered for the American Camp Association's annual tri-state area convention. She reminisced about the kind of bipartisanship that seems to elude Washington today.\"It used to be in Washington you could not escape your adversaries on the political other side because you were always together,\" she said on stage to Jay Jacobs, a camp owner and fellow prominent Democrat from New York, as she recalled her own time as a senator. \"I realized that I might be opposed to somebody's bill today and then working with that person tomorrow.\" Clinton, who recently took to Twitter to express her frustration about Republicans on Capitol Hill, from the letter written by GOP senatorsto Iran to the recent budget proposal, called for more \"relationship building\" between elected officials. \"I've said many times that people who claim proudly never to compromise should not be in the Congress of the United States because I don't think I or anybody have all the answers,\" she said. \"I think we can actually learn things from each other, novel idea.\" Her appearance Thursday raised eyebrows when it was first announced by the ACA last year, because of the high fees that the former Secretary of State's speeches can command. The fee for this speech was not disclosed but, for the first time, the ACA sold tickets for front-row, \"premiere seating\" at the event.Clinton also used this speech to talk about the importance of early education and preserving the environment, both aspects of the summer camp experience. Though she never went to \"sleep away\" camp herself, Clinton described what she had learned from her daughter Chelsea's experiences. \"They're often safe havens in the storms that blow across everyone's life,\" she said, \"places where people can get back to basics and remember or learn for the first time what's really important.\"She added: \"Our families today come in all sizes and shapes "
#inputText = "I am good "
print(inputText)
inputMatrix = getSentenceMatrix(inputText)

predictedSentiment = sess.run(prediction, {input_data: inputMatrix})[0]
#print(prediction)
if (predictedSentiment[0] > predictedSentiment[1]):
    print "Serious News Article"
else:
    print "Satire News Article"
print("\n\n")

#inputText = "Additionally , 72 percent of respondents said they enjoyed the flexibility of home acting versus watching movies in the #theater , noting that if one of them needed to use the restroom or get more popcorn during a pivotal scene , the actors could pause the #performance and do another take when the person got back . Nearly two thirds of those polled also cited a lack of advertisements before #performances as a benefit , as moviegoers were able to jump right into the film without performing any trailers , or only act out the #trailers they were most excited about ."

secinputText = "North Korea escalated its feud with the United States today by declaring it will negotiate with America only if it sends former NBA star Michael Jordan as its representative. The bizarre request will likely add to the already dangerous uncertainty surrounding the intentions of the North Korean regime - which has heavily increased its threats against the U.S. and South Korea in recent weeks. In a statement read live on national television, a pink-clad newswoman announced that a great country like North Korea would only deal with someone of the stature of \"His Airness.\" \"North Korea is a nation of Great People and Great Leaders. We refuse to negotiate with ordinary filth. The evil Western snake will not send its tail to come speak with us - it will send its head. Send us only your Greatest, your most Supreme - Michael 'Air' Jordan.\" The newswoman then threatened \"global destruction\" if the nation's demands were not met in a timely fashion. \"We expect Michael Jordan to be in Pyongyang by midnight Sunday, April 14, or fire from the heavens will rain down upon America and its serpentine allies. \"Explosions will rock the Earth and make all who witness them tremble. Fires will consume your cities until all that remains is ash. Great radioactive clouds will envelop your nation, making it uninhabitable for generations. \"All of this can be a avoided, if Jordan is sent to us in good faith - along with 10,000 pairs of Air Jordan shoes, 10,000 signed Jordan jerseys, and 10,000 bottles of Michael Jordan cologne.\" The North Korean leadership is well known for its love of American basketball, and especially the Michael Jordan-era Chicago Bulls. Former Bulls star Denis Rodman made a trip to North Korea last month in which he personally met with Kim Jong-Un. A spokesperson for the State Department says the United States is \"absolutely not\" considering sending Jordan as its negotiator. However, one official admits it would have a certain logic. \"There aren\'t many people in the world as arrogant as Kim Jong-Un,\" he confides,  \"and Michael Jordan is certainly one of them. They might just hit it off and avoid this whole war.\""

print(secinputText)
inputMatrix = getSentenceMatrix(secinputText)
predictedSentiment = sess.run(prediction, {input_data: inputMatrix})[0]
# predictedSentiment[0] represents output score for positive sentiment
# predictedSentiment[1] represents output score for negative sentiment

if (predictedSentiment[0] > predictedSentiment[1]):
    print "Serious News Article"
else:
    print "Satire News Article"
print("\n")


from os import listdir
from os.path import isfile, join
path_ser = "/home/saad/NewsArticlesAnalysis/SeriousArticlesTest"
path_sat = "/home/saad/NewsArticlesAnalysis/SatireArticlesTest"
onlyfiles_ser = [f for f in listdir(path_ser) if isfile(join(path_ser, f))]
for f in onlyfiles_ser:
	print(f)
