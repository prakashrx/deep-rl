import gym
import tensorflow as tf
import numpy as np
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from ddpg import DDPG

env_name ='CartPole-v1'
env = gym.make(env_name)
env._max_episode_steps = 200

stop_train_score=200 #stop training after reaching score for 3 consecutive episodes
sess = tf.Session()
critic = CriticNetwork(sess, 4, 2, 0.01, 0.001)
actor = ActorNetwork(sess, 4, 2, 0.01, 0.001, activation='softmax')
ddpg = DDPG(sess, actor, critic, batch_size=32)

def train_game( max_steps=10000):
    state = env.reset()
    done = False
    r = 0
    step_count = 0
    while not done and step_count <= max_steps:
        a = ddpg.get_action_for_state(state, True, [0.6, 0.6], [0.5,0.5], [0.2,0.2])
        new_state, reward, done, _ = env.step(np.argmax(a))
        ddpg.step(state, a, reward, new_state, done)
        r += reward
        step_count += 1
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
        new_state, reward, done, _ = env.step(np.argmax(a))
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
        if np.sum(rewards[-3:]) >= stop_train_score * 3:
            break 
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

