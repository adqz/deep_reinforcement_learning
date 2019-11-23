import numpy as np

# fifo buffer with random sample implementation
class ReplayBuffer:
    '''
    Container which stores observations from environment and provide ability to sample and insert more examples
    '''

    def __init__(self, buffer_size, initial_size):
        self.buffer_size = buffer_size
        self.initial_size = initial_size

        self.buffer = []
        self.ready = False

    # return list of tuples (pre_obs, action, reward, obs, done)
    def sample(self, batch_size):
        ind = np.random.choice(range(len(self.buffer)), batch_size, replace=False)
        batch = [self.buffer[i] for i in ind]

        pre_obs = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([[b[2]] for b in batch])
        obs = np.array([b[3] for b in batch])
        done = np.array([b[4] for b in batch])

        return pre_obs, actions, rewards, obs, done

    # insert one tuple into buffer
    def insert(self, tuple):
        # can't use buffer until initial size is met
        if not self.ready: self.ready = len(self.buffer) >= self.initial_size

        # maxed out buffer, so remove first element
        if len(self.buffer) == self.buffer_size: self.buffer.remove(self.buffer[0])

        self.buffer.append(tuple)