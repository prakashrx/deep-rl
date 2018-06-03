import numpy as np
import keras.backend as K
from ReplayBuffer import ReplayBuffer

class DDPG(object):
    
    def __init__(self, sess, actor, critic, buffer_size = 100000, explore=10000, batch_size=32, gamma=0.99, tau = 0.001, lra=1e-3, lrc=1e-2):
        self.sess = sess
        K.set_session(sess)
        
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lra = lra
        self.lrc= lrc
        
        self.epsilon = 1
        self.EXPLORE = explore
        self.actor = actor
        self.critic = critic
        
        self.buffer = ReplayBuffer(self.buffer_size)
        self.mean_loss = 0
        self.steps = 0
        
    
    def get_action_for_state(self, state, noise=False, theta=None, mu=None, sigma=None):
        a_t_original = self.actor.model.predict(state.reshape(-1,state.shape[0]))
        noise_t = np.zeros(a_t_original.shape)
        if noise and self.epsilon > 0:
            a_t_original = np.reshape(a_t_original, (-1,1))
            t = np.reshape(theta, (-1,1))
            m = np.reshape(mu, (-1,1))
            s = np.reshape(sigma, (-1,1))
            noise_t = self.ob(a_t_original, m, t, s)
        return (a_t_original + noise_t).flatten()

    #Ornstein-Uhlenbeck is a stochastic process which has mean-reverting properties.
    # theta * (mu - x) + sigma * np.random.randn(1) 
    # thetaθ means the how “fast” the variable reverts towards to the mean. 
    # mu represents the equilibrium or mean value. 
    # sigma is the degree of volatility of the process. 
    def ob(self, x, mu , theta, sigma):
        return np.multiply(theta, (mu - x)) + np.multiply(sigma,np.random.randn(*sigma.shape))

    def step(self, state, action, reward, new_state, done):
        
        size = self.buffer.add(state,action,reward, new_state, done)
        if(size < self.batch_size):
            return 0
        batch = self.buffer.getbatch(self.batch_size)
        
        states = np.asarray([e[0] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states = np.asarray([e[3] for e in batch])
        dones = np.asarray([e[4] for e in batch])
        y_t = np.asarray([e[1] for e in batch])
        
        target_q_values = self.critic.target.predict([new_states, self.actor.target.predict(new_states)])
        for k in range(len(batch)):
            if dones[k]:
                y_t[k] = rewards[k]
            else:
                y_t[k] = rewards[k] + self.gamma * target_q_values[k]
        
        loss = self.critic.model.train_on_batch([states, actions], y_t)
        self.steps += 1
        self.mean_loss = (1- (1/self.steps)) * self.mean_loss + (1/self.steps) * loss
        self.epsilon -= 1.0 / self.EXPLORE

        a_for_grad = self.actor.model.predict(states)
        grads = self.critic.gradients(states, a_for_grad)
        self.actor.train(states, grads)
        self.actor.train_target_network()
        self.critic.train_target_network()