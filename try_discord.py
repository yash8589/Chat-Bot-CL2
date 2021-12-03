import discord
import os
from discord import Embed
from discord import colour
import requests
import random
import json
import datetime
from datetime import date

client = discord.Client()

import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import pickle

# Preprocessing of data
# we use try and except so we don't have to run the preprocessing and training data part whenever we run the code
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
with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)

# Tensorflow part
from tensorflow.python.framework import ops

ops.reset_default_graph()
# reset tensorflow to remove all previous data

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
model.save("./model.tflearn")


# Predictions
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat(inp):
    # print("\nStart talking with the bot (type quit to stop)!")

    results = model.predict([bag_of_words(inp, words)])[0]
    results_index = numpy.argmax(results)
    tag = labels[results_index]

    if results[results_index] > 0.7:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        return random.choice(responses)
    else:
        msg = "I didn't get that, try again!!"
        return msg


@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.content:
        msg = chat(message.content)
        await message.channel.send(msg)


client.run("")  # put ur discord token inn ""
