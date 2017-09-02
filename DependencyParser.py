import collections
import tensorflow as tf
import numpy as np
import pickle
import math
from progressbar import ProgressBar

from DependencyTree import DependencyTree
from ParsingSystem import ParsingSystem
from Configuration import Configuration
import Config
import Util


"""
This script defines a transition-based dependency parser which makes
use of a classifier powered by a neural network. The neural network
accepts distributed representation inputs: dense, continuous
representations of words, their part of speech tags, and the labels
which connect words in a partial dependency parse.

This is an implementation of the method described in

Danqi Chen and Christopher Manning. A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

Author: Danqi Chen, Jon Gauthier
Modified by: Heeyoung Kwon
"""


wordDict = {}
posDict = {}
labelDict = {}
system = None

def genDictionaries(sents, trees):
    """
    Generate Dictionaries for word, pos, and arc_label
    Since we will use same embedding array for all three groups,
    each element will have unique ID
    """
    word = []
    pos = []
    label = []
    for s in sents:
        for token in s:
            word.append(token['word'])
            pos.append(token['POS'])

    rootLabel = None
    for tree in trees:
        for k in range(1, tree.n+1):
            if tree.getHead(k) == 0:
                rootLabel = tree.getLabel(k)
            else:
                label.append(tree.getLabel(k))

    if rootLabel in label:
        label.remove(rootLabel)

    index = 0
    wordCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    wordCount.extend(collections.Counter(word))
    for word in wordCount:
        wordDict[word] = index
        index += 1

    posCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    posCount.extend(collections.Counter(pos))
    for pos in posCount:
        posDict[pos] = index
        index += 1

    labelCount = [Config.NULL]
    labelCount.extend(collections.Counter(label))
    for label in labelCount:
        labelDict[label] = index
        index += 1

def getWordID(s):
    if s in wordDict:
        return wordDict[s]
    else:
        return wordDict[Config.UNKNOWN]

def getPosID(s):
    if s in posDict:
        return posDict[s]
    else:
        return posDict[Config.UNKNOWN]

def getLabelID(s):
    if s in labelDict:
        return labelDict[s]
    else:
        return labelDict[Config.UNKNOWN]

def getFeatures(c):

    """
    =================================================================

    Implement feature extraction described in
    "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

    =================================================================
    """
    fWord = []
    fPos = []
    fLabel = []
    for j in range(2,-1,-1):
      index = c.getStack(j);
      fWord.append(getWordID(c.getWord(index)))
      fPos.append(getPosID(c.getPOS(index)))
    
    for j in range(0,3):
      index = c.getBuffer(j);
      fWord.append(getWordID(c.getWord(index)))
      fPos.append(getPosID(c.getPOS(index)))

    for j in range(0,2):
      k = c.getStack(j);
      index = c.getLeftChild(k,1);
      fWord.append(getWordID(c.getWord(index)));
      fPos.append(getPosID(c.getPOS(index)));
      fLabel.append(getLabelID(c.getLabel(index)));

      index = c.getRightChild(k,1);
      fWord.append(getWordID(c.getWord(index)));
      fPos.append(getPosID(c.getPOS(index)));
      fLabel.append(getLabelID(c.getLabel(index)));

      index = c.getLeftChild(k, 2);
      fWord.append(getWordID(c.getWord(index)));
      fPos.append(getPosID(c.getPOS(index)));
      fLabel.append(getLabelID(c.getLabel(index)));

      index = c.getRightChild(k, 2);
      fWord.append(getWordID(c.getWord(index)));
      fPos.append(getPosID(c.getPOS(index)));
      fLabel.append(getLabelID(c.getLabel(index)));

      index = c.getLeftChild(c.getLeftChild(k,1),1);
      fWord.append(getWordID(c.getWord(index)));
      fPos.append(getPosID(c.getPOS(index)));
      fLabel.append(getLabelID(c.getLabel(index)));

      index = c.getRightChild(c.getRightChild(k,1),1);
      fWord.append(getWordID(c.getWord(index)));
      fPos.append(getPosID(c.getPOS(index)));
      fLabel.append(getLabelID(c.getLabel(index)));



    feature = []
    feature.extend(fWord);
    #print(len(feature))
    feature.extend(fPos);
    #print(len(feature))
    feature.extend(fLabel);
    #print(len(feature))
    return feature;

def genTrainExamples(sents, trees):
    """
    Generate train examples
    Each configuration of dependency parsing will give us one training instance
    Each instance will contains:
        WordID, PosID, LabelID as described in the paper(Total 48 IDs)
        Label for each arc label:
            correct ones as 1,
            appliable ones as 0,
            non-appliable ones as -1
    """
    numTrans = system.numTransitions()

    features = []
    labels = []
    pbar = ProgressBar()
    for i in pbar(range(len(sents))):
        if trees[i].isProjective():
            c = system.initialConfiguration(sents[i])

            while not system.isTerminal(c):
                oracle = system.getOracle(c, trees[i])
                feat = getFeatures(c)
                #print (type(feat))
                #print (len(feat))
                #print (feat)
                label = []
                for j in range(numTrans):
                    t = system.transitions[j]
                    if t == oracle: label.append(1.)
                    elif system.canApply(c, t): label.append(0.)
                    else: label.append(-1.)

                # print(label)
                features.append(feat)
                labels.append(label)
                c = system.apply(c, oracle)
    return features, labels


#Forward pass with different activation functions
def forward_pass(X, w1, w2, biases):
    ew = tf.matmul(X, w1)
    #layer_1 = tf.nn.sigmoid(tf.add(ew,biases['b1']))
    #layer_1 = tf.nn.relu(tf.add(ew,biases['b1']))
    #layer_1 = tf.nn.tanh(tf.add(ew,biases['b1']))
    layer_1 = tf.pow(tf.add(ew,biases['b1']),3)
    out_layer = tf.matmul(layer_1,w2)
    return out_layer


#Forward pass for 2 hidden layers
def forward_pass_2hidden(embeddings, weights, biases):
    ew1 = tf.matmul(embeddings, weights['w1']) + biases['b1']
    h1 = tf.pow(tf.add(ew1,biases['b1']),3)

    ew2 = tf.matmul(h1, weights['w2']) + biases['b2']
    h2 = tf.pow(tf.add(ew2,biases['b2']),3)

    out_layer = tf.matmul(h2, weights['out'])

    return out_layer


#Activation for parallel cubic activation function
    
# def forward_pass_parallel(X, w1, w2, biases):
#     #Slicing Embed and Weigh Matrices to get parallel cubic activation function
#     X1, X2, X3 = tf.split(X,[18*Config.embedding_size,18*Config.embedding_size,12*Config.embedding_size],1)
#     w11, w12 , w13 = tf.split(w1,[18*Config.embedding_size,18*Config.embedding_size,12*Config.embedding_size],0)
    
#     A = pow(tf.matmul(X1,w11),3)
#     B = pow(tf.matmul(X2,w12),3)
#     C = pow(tf.matmul(X3,w13),3)


#     layer_1 = tf.add(A,B)
#     layer_1 = tf.add(layer_1,C)
#     layer_1 = tf.add(layer_1,biases['b1'])
#     out_layer = tf.matmul(layer_1,w2)
#     return out_layer


if __name__ == '__main__':

    # Load all dataset
    trainSents, trainTrees = Util.loadConll('train.conll')
    devSents, devTrees = Util.loadConll('dev.conll')
    testSents, _ = Util.loadConll('test.conll')
    genDictionaries(trainSents, trainTrees)

    # Load pre-trained word embeddings
    dictionary = {}
    word_embeds = {}
    dictionary, word_embeds = pickle.load(open('word2vec_50.model', 'rb'))
    # Create embedding array for word + pos + arc_label
    embedding_array = np.zeros((len(wordDict) + len(posDict) + len(labelDict), Config.embedding_size))
    len(embedding_array)
    knownWords = list(wordDict.keys())
    print (knownWords[0])
    foundEmbed = 0
    for i in range(len(embedding_array)):
        index = -1
        if i < len(knownWords):
            w = knownWords[i]
            if w in dictionary: index = dictionary[w]
            elif w.lower() in dictionary: index = dictionary[w.lower()]
        if index >= 0:
            #print(dictionary[knownWords[i].lower()])
            foundEmbed += 1
            embedding_array[i] = word_embeds[index]
        else:
            embedding_array[i] = np.random.rand(Config.embedding_size)*0.02-0.01
    print ("Found embeddings: ", foundEmbed, "/", len(knownWords))

    # Get a new instance of ParsingSystem with arc_labels
    print(len(list(labelDict.keys())))
    print(len(list(wordDict.keys())))
    print(len(list(posDict.keys())))
    system = ParsingSystem(list(labelDict.keys()))

    print ("Generating Traning Examples")
    trainFeats, trainLabels = genTrainExamples(trainSents, trainTrees)
    pickle.dump([trainFeats, trainLabels], open('data.model', 'wb'))
    
    #trainFeats, trainLabels = pickle.load(open('data.model', 'rb'))
    

    graph = tf.Graph()

    with graph.as_default():

        #To fix input embeddings
        #embeddings = tf.constant(embedding_array, dtype=tf.float32,trainable=False)
        embeddings = tf.constant(embedding_array, dtype=tf.float32)

        train_inputs = tf.placeholder(tf.int32, shape=[Config.batch_size,Config.n_Tokens])
        train_labels = tf.placeholder(tf.int32, shape=[Config.batch_size,system.numTransitions()])
        test_inputs = tf.placeholder(tf.int32, shape=[Config.n_Tokens,])
        

        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        embed_new = tf.reshape(embed,[Config.batch_size,Config.n_Tokens*Config.embedding_size])
        embed_test = tf.nn.embedding_lookup(embeddings, test_inputs)
        embed_test = tf.reshape(embed_test,[1,Config.n_Tokens*Config.embedding_size])
        
        #Trying different weights

#             weights = {
#     'w1': tf.Variable(tf.random_normal([Config.n_Tokens*Config.embedding_size,Config.hidden_size])),
#     'w2': tf.Variable(tf.random_normal([Config.hidden_size,91]))
# }
        #     weights = {

        #     'w1': tf.Variable(
        #     tf.truncated_normal([Config.n_Tokens*Config.embedding_size, Config.hidden_size],
        #                     stddev=1.0 / math.sqrt(Config.hidden_size))),
        #     'w2': tf.Variable(
        #     tf.truncated_normal([Config.hidden_size,system.numTransitions()],
        #                     stddev=1.0 / math.sqrt(system.numTransitions())))
        # }

        #Weights for 1 hidden layer
        weights = {
            'w1': tf.Variable(tf.truncated_normal([Config.embedding_size * Config.n_Tokens, Config.hidden_size], stddev=1.0 / math.sqrt(Config.n_Tokens))),
            'out': tf.Variable(tf.truncated_normal([Config.hidden_size, system.numTransitions()], stddev=1.0 / math.sqrt(Config.n_Tokens)))
        }

        #Weights for 2 hidden layers
        # weights = {
        #     'w1': tf.Variable(tf.random_normal([Config.embedding_size * Config.n_Tokens, Config.hidden_size])),
        #     'w2': tf.Variable(tf.random_normal([Config.hidden_size, Config.hidden_size2])),
        #     'out': tf.Variable(tf.random_normal([Config.hidden_size2, system.numTransitions()]))
        # }

        biases = {
            'b1': tf.Variable(tf.zeros([Config.hidden_size]))
            #'b2': tf.Variable(tf.zeros([Config.hidden_size2]))
        }
      
        y_hat = forward_pass(embed_new,weights['w1'],weights['out'],biases)
        #y_hat = forward_pass_2hidden(embed_new,weights,biases)
        temp_labels = tf.arg_max(train_labels,dimension = 1)

        test_pred = forward_pass(embed_test,weights['w1'],weights['out'],biases)
        #test_pred = forward_pass_2hidden(embed_test,weights,biases)
        test_pred = tf.nn.softmax(test_pred)


        #loss = -1*tf.reduce_mean(tf_func.cross_entropy_loss(y_hat,train_labels)) 
        #+ Config.lam*0.5*(tf.nn.l2_loss(weights['w1']) + tf.nn.l2_loss(weights['w2']) + tf.nn.l2_loss(biases['b1']) +tf.nn.l2_loss(embed_new))
      
        #Loss for 1 hidden layer
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_hat, labels=temp_labels) 
        + Config.lam*(tf.nn.l2_loss(weights['w1']) + tf.nn.l2_loss(weights['out']) + tf.nn.l2_loss(biases['b1']) +tf.nn.l2_loss(embed_new)))

        #Loss for 2 hidden Layers
        # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_hat, labels=temp_labels) 
        # + Config.lam*0.5*(tf.nn.l2_loss(weights['w1']) + tf.nn.l2_loss(weights['w2']) + tf.nn.l2_loss(weights['out'])  +tf.nn.l2_loss(embed_new)))


        """
        ===================================================================

        Define the computational graph with necessary variables.
        You may need placeholders of:
            train_inputs
            train_labels
            test_inputs

        Implement the loss function described in the paper

        ===================================================================
        """

        optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)

        #Compute Gradients
        grads = optimizer.compute_gradients(loss)
        # Gradient Clipping
        clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
        app = optimizer.apply_gradients(clipped_grads)


        init = tf.global_variables_initializer()

    num_steps = Config.max_iter
    with tf.Session(graph=graph) as sess:
        init.run()
        print ("Initialized")

        average_loss = 0
        for step in range(num_steps):
            start = (step*Config.batch_size)%len(trainFeats)
            end = ((step+1)*Config.batch_size)%len(trainFeats)
            if end < start:
                start -= end
                end = len(trainFeats)
            batch_inputs, batch_labels = trainFeats[start:end], trainLabels[start:end]

            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            _,loss_val = sess.run([app,loss],feed_dict=feed_dict)
            average_loss += loss_val
            # print("Y Hat")
            # print(output)
            # #print(loss_val)
            # print("weights")
            # print(wts)
            # print("embedings")
            # print(emb)
            # Display average loss
            if step % Config.display_step == 0:
                if step > 0:
                    average_loss /= Config.display_step
                print ("Average loss at step ", step, ": ", average_loss)
                average_loss = 0

            # Print out the performance on dev set
            if step % Config.validation_step == 0 and step != 0:
                print ("\nTesting on dev set at step ", step)
                predTrees = []
                for sent in devSents:
                    numTrans = system.numTransitions()

                    c = system.initialConfiguration(sent)
                    while not system.isTerminal(c):
                        feat = getFeatures(c)
                        pred = sess.run(test_pred, feed_dict={test_inputs: feat})

                        optScore = -float('inf')
                        optTrans = ""

                        for j in range(numTrans):
                            if pred[0, j] > optScore and system.canApply(c, system.transitions[j]):
                                optScore = pred[0, j]
                                optTrans = system.transitions[j]

                        c = system.apply(c, optTrans)

                    predTrees.append(c.tree)
                result = system.evaluate(devSents, predTrees, devTrees)
                print (result)

        print ("Optimization Finished.")

        print ("Start predicting on test set")
        predTrees = []
        for sent in testSents:
            numTrans = system.numTransitions()

            c = system.initialConfiguration(sent)
            while not system.isTerminal(c):
                feat = getFeatures(c)
                pred = sess.run(test_pred, feed_dict={test_inputs: feat})

                optScore = -float('inf')
                optTrans = ""

                for j in range(numTrans):
                    if pred[0, j] > optScore and system.canApply(c, system.transitions[j]):
                        optScore = pred[0, j]
                        optTrans = system.transitions[j]

                c = system.apply(c, optTrans)

            predTrees.append(c.tree)
        print ("Store the test results.")
        Util.writeConll('result.conll', testSents, predTrees)

