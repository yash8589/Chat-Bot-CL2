import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json

# Preprocessing of data
with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))  # set removes duplicates from words

labels = sorted(labels)

# start training of data
training = []
output = []

# convert list into a bag of words - 1 Hot encoding
out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []  # bag of words

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:  # word exists in the current pattern
            bag.append(1)
        else:  # word isn't here
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

# Tensorflow part
from tensorflow.python.framework import ops

ops.reset_default_graph()
# tensorflow.reset_default_graph(
# )  # reset tensorflow to remove all previous data

net = tflearn.input_data(shape=[None, len(
    training[0])])  # Define input shape that we are expecting from our model
net = tflearn.fully_connected(
    net, 8
)  # add a fully connected layer to our neural network with 8 neurons which starts at the previous input data (Hidden Layer)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(
    net, len(output[0]), activation="softmax"
)  # get probability of each neuron in the layer using softnmax
net = tflearn.regression(net)

model = tflearn.DNN(net)  # Type of Neural network

model.fit(
    training, output, n_epoch=1000, batch_size=8, show_metric=True
)  # number of epoch is the amount of times it is going to see the same data
model.save("model.tflearn")