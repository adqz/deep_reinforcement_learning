import numpy as np

# fifo buffer with random sample implementation
class ReplayBuffer:
    '''
    Container which stores experiences and provides ability to 
    1. Sample examples
    2. Insert example

    An example is a five element tuple as follows: (pre_obs, action, reward, obs, done)

    The attribute "buffer" must be used to access contents of the buffer. It is of type list.

    The buffer loosely follows the queue data structure.
    If the buffer gets full and a new sample needs to be inserted, 
    the entry at index 0 gets removed (least recent) to accomodate the new entry.

    Dependencies - numpy
    For more details on buffer refer - https://tinyurl.com/t2ur26m
    '''

    def __init__(self, buffer_size, initial_size):
        '''
        Initialize buffer and specify total buffer size and how many examples to initialize it with
        
        :param int buffer_size: Total bufffer size
        :param int initial_size: Number of examples to initialize buffer with. Actions are chosen at random.
        :return: None
        :rtype: None
        :raises AssertionError: If buffer_size and initial_size are not of type int
        '''
        assert isinstance(buffer_size, int) and buffer_size>0
        assert isinstance(initial_size, int) and initial_size>0

        self.buffer_size = buffer_size
        self.initial_size = initial_size

        self.buffer = []
        self.ready = False

    # return list of tuples (pre_obs, action, reward, obs, done)
    def sample(self, batch_size):
        '''
        Returns a batch of size batch_size from the buffer.
        :param int batch_size: Number of examples to sample from buffer
        :returns: Tuple containing five numpy arrays, each of size (batch_size,) in the order - (pre_obs, action, reward, obs, done)
        :rtype: tuple
        :raises AssertionError: If batch_size is not in range [initial_size, buffer_size]
        '''
        assert isinstance(batch_size, int)
        assert self.initial_size <= batch_size <= self.buffer_size

        ind = np.random.choice(range(len(self.buffer)), batch_size, replace=False)
        batch = [self.buffer[i] for i in ind]

        pre_obs = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([[b[2]] for b in batch])
        obs = np.array([b[3] for b in batch])
        done = np.array([b[4] for b in batch])

        return pre_obs, actions, rewards, obs, done

    # insert one tuple into buffer
    def insert(self, example):
        '''
        Insert a single example in to the buffer. If buffer is full, remove the example at index 0
        to accomodate the new example. 

        :param tuple example: New example to add to buffer. Format should be in the following order: (pre_obs, action, reward, obs, done)
        :return: None
        :rtype: None
        :raises AssertionError: If input is not tuple and is not of length five
        '''
        assert isinstance(example, tuple) and len(example)==5

        # can't use buffer until initial size is met
        if not self.ready: self.ready = len(self.buffer) >= self.initial_size

        # maxed out buffer, so remove first element
        if len(self.buffer) == self.buffer_size: self.buffer.remove(self.buffer[0])

        self.buffer.append(example)