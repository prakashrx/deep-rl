from collections import deque
import random

class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque()
    
    def getbatch(self, size):
        buffer_size = size if len(self.buffer) >= size else len(self.buffer)
        return random.sample(self.buffer, buffer_size)
    
    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if len(self.buffer) >= self.buffer_size:
            self.buffer.popleft()
        self.buffer.append(experience)
        return len(self.buffer)
        
    def clear(self):
        self.buffer = deque()