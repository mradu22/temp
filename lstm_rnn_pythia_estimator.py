import numpy as np
import datetime, time
import tensorflow as tf
import sys, os, re, gzip, json
from random import randint
from sklearn.utils import shuffle

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default = 128, type = int, help ='batch size')
parser.add_argument("--train_steps", default = 50, type = int, help ='number of training steps')
parser.add_argument("--lstm_units", default = 64, type = int, help = "number of lstm units in each cell")

######################INPUT CUSTOM SENTENCE###########################
#define cleaning functions
# Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
import re
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

maxSeqLength = 400

def cleanSentences(string_list):
    """input a list of sentences here"""
    string_list = [string.lower().replace("<br />", " ") for string in string_list]
    return [re.sub(strip_special_chars, "", string.lower()) for string in string_list]

def getSentenceMatrix(sentences):
    arr = np.zeros([batch_size, maxSeqLength])
    sentenceMatrix = np.zeros([batchSize,maxSeqLength], dtype='int32')
    cleanedSentences = cleanSentences(sentences)
    for i in range(len(cleanedSentences)):
        split = cleanedSentences[i].split()
        for indexCounter,word in enumerate(split):
            try:
                sentenceMatrix[i,indexCounter] = wordsList.index(word)
            except ValueError:
                sentenceMatrix[i,indexCounter] = 399999 #Vector for unkown words
    return sentenceMatrix
#########################################################################


def loadz(fname):
    #it will load a json.gs file from the working folder
    #of your project. fname is the file name without the
    #extension. returns the data in the file
    with gzip.GzipFile(fname + '.json.gz', 'r') as fin:    # 4. gzip
        json_bytes = fin.read()                         # 3. bytes (i.e. UTF-8)
        json_str = json_bytes.decode('utf-8')            # 2. string (i.e. JSON)
        data = json.loads(json_str)                      # 1. data
    return data

def savez(data,fname):
    #saves a list ( data ) into a json.gz file in the working folder
    #of your project. fname is the file name without the
    #extension.
    json_str = json.dumps(data) + "\n"               # 2. string (i.e. JSON)
    json_bytes = json_str.encode('utf-8')            # 3. bytes (i.e. UTF-8)
    with gzip.GzipFile(fname + '.json.gz', 'w') as fout:   # 4. gzip
        fout.write(json_bytes)
    print('Saved as {}'.format(fname + '.json.gz'))

def read_dataset(file_name = 'pythia_dataset', dev_size=20000, test_size=20000):
    dataset = loadz(file_name)
    #put it into a numpy.array
    dataset = shuffle(np.array(dataset[0]), np.array(dataset[1]))
    features_train, features_dev, features_test = np.split(dataset[0], [-(test_size+dev_size), -test_size])
    labels_train, labels_dev, labels_test = np.split(dataset[1], [-(test_size+dev_size), -test_size])
    return ((features_train, labels_train), (features_dev, labels_dev), (features_test, labels_test))

def predict_input_fn():
    wordsList = np.load('wordsList.npy')
    print('Loaded word list.')
    wordsList = wordsList.tolist() #loaded as numpy array now is a list
    wordsList = [word.decode('UTF-8') for word in wordsList] #encode as UTF8

    return


def my_model(features, labels, mode, params):
    """LSTM with one layer of 64 units."""
    X = features
    Y = labels
    lstmUnits = 64
    numClasses = 2

    wordVectors = np.load('wordVectors.npy')
    print('Loaded word vectors.')
    data = tf.nn.embedding_lookup(wordVectors,X)
    #lstm cells
    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    #lstmCell = tf.contrib.rnn.DropoutWrapper(cell = lstmCell, output_keep_prob = 0.75)
    value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype = tf.float32)

    #weights and bias
    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
    bias = tf.Variable(tf.constant(0.1, shape = [numClasses]))
    value = tf.transpose(value, [1,0,2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)

    #prediction
    prediction = (tf.matmul(last,weight) + bias)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = prediction, labels = Y))

    if mode == tf.estimator.ModeKeys.PREDICT:

        predictions = {
        "class_ids": preds,
        "probabilities": tf.nn.softmax(prediction),
        "logits": prediction
        }
        return tf.estimator.EstimatorSpec(mode, predictions = predictions)

    if mode == tf.estimator.ModeKeys.EVAL:
        preds = tf.argmax(prediction, 1)
        actuals = tf.argmax(Y, 1)
        accuracy = tf.metrics.accuracy(labels = actuals,
                                   predictions = preds,
                                   name='acc_op')
        precision = tf.metrics.precision(labels = actuals,
                                   predictions = preds,
                                   name='acc_op')
        recall = tf.metrics.recall(labels = actuals,
                                   predictions = preds,
                                   name='acc_op')

        metrics = {'accuracy': accuracy,'precision': precision,'recall': recall}
        return tf.estimator.EstimatorSpec(mode, loss=loss,  eval_metric_ops = metrics)

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.train.AdamOptimizer().minimize(loss, global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss = loss, train_op = train_op)

def input_fn(features, labels, batch_size, is_training):
    def _input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((features,labels))

        if is_training:
            dataset = dataset.shuffle(100000).repeat()

        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        return dataset.make_one_shot_iterator().get_next()
    return _input_fn



def test_input_fn(features, labels, batch_size):
    #same for test
    test_data = tf.data.Dataset.from_tensor_slices(test)
    test_data = test_data.shuffle(10000)
    #test_data = test_data.repeat()
    test_data = test_data.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    return dataset.make_one_shot_iterator().get_next()

def main(argv):
    args = parser.parse_args(argv[1:])
    print('batch size:',args.batch_size)
    print('train steps:',args.train_steps)

    #model parameters
    maxSeqLength = 400 #number of words in input sequnce. if longer we will padd with zeros
    numDimensions = 50 #embedding dimension for each word

    print('Max words in input sequence (anything longer padded with zeros): {}\nWord embedding dimension: {}\n'.format(maxSeqLength,numDimensions))

    t0=time.time()
    print('Loading dataset... ', end='')
    # train, dev ,test = loadz('pythia_train'), loadz('pythia_dev'), loadz('pythia_test')
    # you still need to turn everything into numpy arrays after doigng this thing
    #for temporary use whily you build the graph
    train, dev, test = read_dataset('pythia_dev', 1000, 500)
    print('completed in {} seconds. '.format(time.time()-t0))

    numTrain = train[0].shape[0]
    numTest = test[0].shape[0]
    numDev = dev[0].shape[0]
    print('Train/dev/test split: {}/{}/{}'.format(numTrain,numDev,numTest))

    pythia_classifier = tf.estimator.Estimator(model_fn = my_model)

    for _ in range(10):
        train_input_fn = input_fn(train[0],train[1],args.batch_size, is_training = True)
        pythia_classifier.train(input_fn = train_input_fn, steps=args.train_steps)

        # Evaluate the model.
        test_input_fn = input_fn(test[0],test[1],args.batch_size, is_training = False)
        eval_result = pythia_classifier.evaluate(input_fn=test_input_fn)

        print('\nTest set accuracy: {accuracy:0.3f}\nTest set precision: {precision:0.3f}\nTest set recall: {recall:0.3f}'.format(**eval_result))






if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
