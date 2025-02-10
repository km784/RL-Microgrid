
import numpy as np



def eGreedy(Q, marker, epsilon):
    r = np.random.rand()  # generate a random number

    if marker == 1:  # control damping lies on the lower boundary
        if r > epsilon:
            v, a = np.max(Q[1:3]), np.argmax(Q[1:3]) + 1  # value, action
        else:
            a = np.random.randint(1, 3) + 1  # random integer based on a uniform distribution
    elif marker == 2:  # control damping lies on the upper boundary
        if r > epsilon:
            v, a = np.max(Q[0:2]), np.argmax(Q[0:2])  # value, action
        else:
            a = np.random.randint(0, 2)  # random integer based on a uniform distribution
    else:
        if r > epsilon:
            v, a = np.max(Q), np.argmax(Q)  # value, action
        else:
            a = np.random.randint(0, 3)  # random integer based on a uniform distribution

    return a
