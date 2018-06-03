import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
import keras.backend as K
import json
import os

H1 = 128
H2 = 64
class ActorNetwork(object):
    
    
    def __init__(self, sess, state_size, action_size, tau, lr, activation='sigmoid'):
        
        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size
        self.tau = tau
        self.lr = lr
        self.activation = activation

        K.set_session(self.sess)
        
        self.model, self.weights, self.state = self.create()
        self.target, self.target_weights, _ = self.create()
        self.action_gradients = tf.placeholder(tf.float64, [None, action_size])
        self.param_grads = tf.gradients(self.model.outputs, self.weights, -self.action_gradients)
        grads = zip(self.param_grads, self.weights)
        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(grads)
        
        self.sess.run(tf.global_variables_initializer())
    
    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradients : action_grads
        })
        
    def train_target_network(self):
        model_weights = self.model.get_weights()
        target_weights = self.target.get_weights()
        for i in range(len(model_weights)):
            target_weights[i] = self.tau*model_weights[i] + (1 - self.tau)*target_weights[i]
        self.target.set_weights(target_weights)
        
    def create(self):
        S = Input(shape=[self.state_size], name='state')
        h0 = Dense(H1, activation='relu')(S)
        h1 = Dense(H2, activation='relu')(h0)
        A =  Dense(self.action_size, activation=self.activation)(h1)
        
        model = Model(inputs=S, outputs= A)
        return model, model.trainable_weights, S
    
    def save(self, filename='actormodel', output_json=False):
        self.model.save_weights(filename + '.h5', overwrite=True)
        if output_json:
            with open(filename + '.json', "w") as outfile:
                json.dump(self.model.to_json(), outfile)

    def load(self, filename='actormodel'):
        if os.path.exists(filename + '.h5'):
            print('loading ' + filename)
            self.model.load_weights(filename + '.h5')
            self.target.load_weights(filename + '.h5')