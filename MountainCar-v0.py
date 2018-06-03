import gym
import tensorflow as tf
import numpy as np
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from ddpg import DDPG
import random
env_name ='MountainCar-v0'
env = gym.make(env_name)

sess = tf.Session()
critic = CriticNetwork(sess, 2, 3, 0.01, 0.001)
actor = ActorNetwork(sess, 2, 3, 0.01, 0.001, activation='softmax')
ddpg = DDPG(sess, actor, critic, batch_size=32, explore=1000000)

def train_game( max_steps=200):
    state = env.reset()
    done = False
    r = 0
    step_count = 0
    actions = [[0.98,0.01,0.01], [0.01,0.98,0.01], [0.01,0.01, 0.98]]
    while not done and step_count <= max_steps:
        if ddpg.epsilon < np.random.random():
            a = ddpg.get_action_for_state(state)
            #inx = np.random.choice([0,1,2], p=actionProb)
            inx = np.argmax(a)
        else:
            inx = np.random.choice([0,1,2])
            a = actions[inx]
        new_state, reward, done, _ = env.step(inx)
        reward = new_state[0]
        ddpg.step(state, a, reward, new_state, done)
        step_count += 1
        r += reward
        state = new_state
        
    return r, ddpg.mean_loss

def play_game( max_steps=1000 ):
    state = env.reset()
    done = False
    r = 0
    step_count = 0
    while not done and step_count <= max_steps:
        env.render()
        a = ddpg.get_action_for_state(state)
        inx = np.argmax(a)
        new_state, reward, done, _ = env.step(inx)
        r += reward
        step_count += 1
        state = new_state
    return r

def train(load_weights=True, episodes = 1000):

    if load_weights:
        actor.load(env_name + '-actor')
        critic.load(env_name + '-critic')

    rewards = []
    for i in range(episodes):
        r, l = train_game()
        rewards.append(r)
        print('episode {0}, loss {1}, reward {2}'.format(i, l, r))

    actor.save(env_name + '-actor')
    critic.save(env_name + '-critic')

def play():
    actor.load(env_name + '-actor')
    critic.load(env_name + '-critic')
    choice = 'y'
    while choice == 'y':
        print(play_game())
        print("Play Again? [y/n]")
        choice = input().lower()

def main():
    print("Train? [y/n]")
    choice = input().lower()
    if choice == 'y':
        train()
        print("Training over. Press any key to start play")
        choice = input().lower()
    play()

    sess.close()

if __name__ == "__main__":
    main()

