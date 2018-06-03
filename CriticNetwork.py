import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Add
from keras.optimizers import Adam
import keras.backend as K
import json
import os
H1 = 128
H2 = 64
class CriticNetwork(object):
    
    def __init__(self, sess, state_size, action_size, tau, lr):
        
        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size
        self.tau = tau
        self.lr = lr
        K.set_session(self.sess)
        self.model, self.state, self.action = self.create()
        self.target, self.target_state, self.target_action = self.create()
        self.action_grads = tf.gradients(self.model.outputs, self.action)
        self.sess.run(tf.global_variables_initializer())
    
    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state : states,
            self.action : actions
        })[0]
    
    def train_target_network(self):
        
        model_weights = self.model.get_weights()
        target_weights = self.target.get_weights()
        
        for i in range(len(model_weights)):
            target_weights[i] = self.tau*model_weights[i] + (1 - self.tau)*target_weights[i]
        
        self.target.set_weights(target_weights)
        
        
    def create(self):

        S = Input(shape=[self.state_size], name='state')
        s1 = Dense(H1, activation='relu')(S)
        s2 = Dense(H2, activation='linear')(s1)
        
        A = Input(shape=[self.action_size], name='action')
        a1 = Dense(H2, activation='linear')(A)
        
        M = Add()([s2,a1])
        m1 = Dense(H2, activation='relu')(M)
        V = Dense(self.action_size, activation='linear')(m1)
        
        model = Model(inputs=[S,A], outputs=V)
        adam = Adam(lr= self.lr)
        model.compile(loss='mean_squared_error', optimizer=adam)
        
        return model, S, A
    
    def save(self, filename='criticmodel', output_json=False):
        self.model.save_weights(filename + '.h5', overwrite=True)
        if output_json:
            with open(filename + '.json', "w") as outfile:
                json.dump(self.model.to_json(), outfile)
        
    def load(self, filename='criticmodel'):
        if os.path.exists(filename + '.h5'):
            print('loading ' + filename)
            self.model.load_weights(filename + '.h5')
            self.target.load_weights(filename + '.h5')