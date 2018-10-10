#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 12:08:02 2017

@authors: Rohit Konda, Vineet Tiruvadi
#Modules for working with dynamical network models
Notes:
    Might be a smart idea to merge with TVB modules if they already do a lot of this
"""

# import statements
import networkx as nx
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import collections as col


# Class  encapsulation of the network model
class nmodel:
    def __init__(self, G, x, h, f, g=None, dt=.05):

        self.G = G  # Graph representation of network
        self.x = np.array([np.array([np.array([j]) for j in i]) for i in np.array(x)])  # states 
        self.h = h  # array of node functions
        self.f = f  # array of coupling functions
        self.g = g  # output function
        self.y = np.array(self.output())  # measurement vectors
        self.t = 0  # time
        self.dt = dt  # time step

        # checks
        if len(self.x) == 0:
            raise ValueError('self.x can\'t have less than one state')
        if len(self.x) != nx.number_of_nodes(self.G):
            raise ValueError('length of self.x must match number of nodes')

    # state derivative
    def dev(self, state):
        if np.iscomplex(state).any():
            dev = np.matrix(np.zeros(np.shape(state), dtype=np.complex_))
        else:
            dev = np.matrix(np.zeros(np.shape(state)))
        c = 0  # counter if f depends on the edge
        for i in range(0, len(state)):
            sumEdge = np.array(dev[i].tolist()[0])
            if self.G[i]:
                for j in self.G[i]:
                    if isinstance(self.f, col.Iterable):
                        if len(self.f) == len(self.x):
                            sumEdge += self.f[i](state[i], state[j])
                        elif len(self.f) == self.G.number_of_edges():
                            sumEdge += self.f[c](state[i], state[j])
                            c += 1
                        else:
                            raise ValueError('length of f must either be equal to the number of nodes or edges')
                    elif callable(self.f):
                        sumEdge += self.f(state[i], state[j])
                    else:
                        raise ValueError('f must be either be iterable or callable')
            if isinstance(self.h, col.Iterable):
                dev[i] = self.h[i](state[i]) + sumEdge
            elif callable(self.h):
                dev[i] = self.h(state[i]) + sumEdge
            else:
                raise ValueError('h must be either be iterable or callable')
        return dev

    # output function
    def output(self):
        if self.g is None:
            return np.matrix(self.x[:, :, -1])
        else:
            return self.g(self.x[:, :, -1])

    # euler method approximation of behavior
    def euler_step(self):
        new_state = self.x[:, :, -1] + self.dev(self.x[:, :, -1])*self.dt
        self.t += self.dt
        self.x = np.dstack((self.x, np.array(new_state)))
        self.y = np.dstack((self.y, np.array(self.output())))

    # runge-Kutta approximation of behavior
    def runge_kutta_step(self):
        k1 = self.dev(self.x[:, :, -1])*self.dt
        k2 = self.dev(np.array(self.x[:, :, -1] + .5*k1))*self.dt
        k3 = self.dev(np.array(self.x[:, :, -1] + .5*k2))*self.dt
        k4 = self.dev(np.array(self.x[:, :, -1] + k3))*self.dt
        new_state = self.x[:, :, -1] + (k1 + 2*k2 + 2*k3 + k4)/6
        self.t += self.dt
        self.x = np.dstack((self.x, np.array(new_state)))
        self.y = np.dstack((self.y, np.array(self.output())))

    # time step function
    def step(self):
        self.runge_kutta_step()

    # runs model for time T and stores states
    def run(self, T):
        for ts in range(0, int(T/self.dt)):
            self.step()

    # clears all states exept initial
    def clear_run(self):
        self.x = np.array([np.array([np.array([int(j)]) for j in i]) for i in self.x[:, :, 0]])
        self.y = self.y[:, :, 0]


# lambda function for creating vectorized distributions
create_vec_states = lambda param: np.concatenate((np.array([create_states(*tup) for tup in param])), axis = 1)


# creates specified states
# n: number of states
# distribution: logistic, normal, uniform, or point
# for point: a = value
# for normal/logistic: a: mean, b: scale(spread)
# for uniform: a:lower bound, b:upper bound
# for complex tuple c: respective "a" and "b" for imaginary parts
def create_states(n, a, b=None, distribution=None, c=(0, 0)):
    if distribution is None:
        if c == (0, 0):
            return np.matrix([a]*n).T
        else:
            return np.matrix([a]*n).T + np.matrix([c[0]]*n).T * 1j
    elif distribution == 'logistic':
        if c == (0, 0):
            return np.matrix(np.random.logistic(a, b, n)).T
        else:
            return np.matrix(np.random.logistic(a, b, n)).T + \
             np.matrix(np.random.logistic(c[0], c[1], n)).T * 1j
    elif distribution == 'normal':
        if c == (0, 0):
            return np.matrix(np.random.normal(a, b, n)).T
        else:
            return np.matrix(np.random.normal(a, b, n)).T + \
             np.matrix(np.random.normal(c[0], c[1], n)).T * 1j
    elif distribution == 'uniform':
        if c == (0, 0):
            return np.matrix(np.random.uniform(a, b, n)).T
        else:
            return np.matrix(np.random.uniform(a, b, n)).T + \
             np.matrix(np.random.uniform(c[0], c[1], n)).T * 1j


# create SCC graph
def create_SCC():
        nodes = [0, 1, 2, 3, 4, 5]
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from([(0, 1), (1, 2), (0, 5), (0, 2), (2, 3), (3, 4), (3, 5), (4, 5)])
        return G


# multiply the input graph by a factor of side
# nodal: probability of connectivity in nodes
# edges: probability of connectivity within edges
# other: probability of connectivity with no edges
def multiply_graph(G, side, nodal=.9, edges=.5, other=.1):
    A = nx.to_numpy_matrix(G)
    B = [1]*(len(A)*side)
    i = 0
    for r in A.tolist():
        row = [[]]*side
        j = 0
        for c in r:
            if c > 0:
                cluster = [[1 if col < edges else 0 for col in row] for row in np.random.rand(side, side)]
            elif j == i:
                cluster = [[1 if col < nodal else 0 for col in row] for row in np.random.rand(side, side)]
            else:
                cluster = [[1 if col < other else 0 for col in row] for row in np.random.rand(side, side)]
            row = np.hstack((row, cluster))
            j += 1
        B = np.vstack((B, row))
        i += 1
    B = B[1:][:]
    for i in range(len(B)):
                B[i][i] = 0
                for j in range(i):
                    B[i][j] = B[j][i]
    return nx.from_numpy_matrix(B)


# plots connections as an adjacency matrix
def plt_graph(G):
    A = nx.to_numpy_matrix(G)
    plt.figure()
    plt.imshow(A, interpolation="nearest")


# plots states
def state_course(states):
    plt.figure()
    plt.plot(states.T)
    plt.xlabel('Time Steps')
    plt.ylabel('x')
    plt.title('t')
    plt.show()


# plots spectogram
def spectrogram(signal, fs=1.0, window=('tukey', 0.25), nperseg=None, noverlap=None, nfft=None):
    f, t, s = sig.spectrogram(signal, fs, window, nperseg, noverlap, nfft)
    plt.pcolormesh(t, f, s)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.show()


# plots PSD
def PSD(signal, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None):
    f, Pxx = sig.welch(signal, fs, window, nperseg, noverlap, nfft)
    plt.semilogy(f, Pxx)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()


# get time courses from specified part of the vector state
reduce_state = lambda i, x:  np.matrix([row[i] for row in x])