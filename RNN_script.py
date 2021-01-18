import sys
from collections import OrderedDict
import matplotlib.pyplot as plt
import random
from math import floor
import numpy as np

#-----------------------------------------------------------------------------
#DATA PREP
#-----------------------------------------------------------------------------

def load_data(filename):
        return open(filename, 'r', encoding='utf8').read()

def get_data_info(book_data):
    book_chars = get_book_chars(book_data)
    vocab_len = get_vocab_len(book_chars)
    char_to_ind = get_char_to_ind(book_chars)
    ind_to_char = get_ind_to_char(book_chars)

    data_info = {"book_data": book_data,
            "book_chars": book_chars,
            "vocab_len": vocab_len,
            "char_to_ind": char_to_ind,
            "ind_to_char": ind_to_char}
    return data_info

def get_book_chars(book_data):
    return list(set(book_data))

def get_vocab_len(book_chars):
    return len(book_chars)

def get_char_to_ind(book_chars):
    return OrderedDict((c, i) for i, c in enumerate(book_chars))

def get_ind_to_char(book_chars):
    return OrderedDict((i, c) for i, c in enumerate(book_chars))

def get_one_hot(inputs, vocab_len, n):
    x = np.zeros((n , vocab_len, 1))
    for t in range(n):
        x[t][inputs[t]] = 1
    return x

def get_one_hot_2(inputs, i, vocab_len):
    x = np.zeros((vocab_len, 1))
    x[i] = 1
    return x

#-----------------------------------------------------------------------------
#HELPER FUN
#-----------------------------------------------------------------------------
    
def _softmax(x):
    s = np.exp(x - np.max(x, axis=0)) / np.exp(x - np.max(x, axis=0)).sum(axis=0)
    return s

def _tanh(x):
    return np.tanh(x)

def _cross_entropy(p, y):
        return -np.log(np.dot(y.T,p))

def check_grad(g1,g2):
        h = 1e-6
        return np.linalg.norm(g1 - g2,ord=1) / max(h ,np.linalg.norm(g1 ,ord=1) + np.linalg.norm(g2 ,ord=2))

def check_gradients(grads_ana, grads_num):
        for grad in range(len(grads_ana)):
            error = check_grad(grads_ana[grad],grads_num[grad])
            print('Gradient: ' + str(grad) + ' has mre: ' + str(error))

#-----------------------------------------------------------------------------
#PLOTS
#-----------------------------------------------------------------------------

def plot_performance(train):
       iter = np.arange(len(train))

       plt.plot(iter, train)
       plt.xlabel('Iteration number times X')
       plt.ylabel('Smooth loss')
       plt.grid()
       plt.savefig("smoothloss" + ".png")
       plt.close()

#-----------------------------------------------------------------------------
#MODEL
#-----------------------------------------------------------------------------

class RNN():
    def __init__(self, data, m=10, eta=.1, seq_length=25):
        self.m = m
        self.eta = eta 
        self.N = seq_length

        self.book_data = data['book_data']
        self.book_chars = data['book_chars']
        self.vocab_len = data['vocab_len']
        self.char_to_ind = data['char_to_ind']
        self.ind_to_char = data['ind_to_char']
            
        self.K = self.vocab_len
        mu = 0
        sigma = 0.01
        self.b = np.zeros((m, 1))
        self.c = np.zeros((self.K, 1))
        self.U = np.random.normal(mu, sigma, size=(m, self.K))
        self.W = np.random.normal(mu, sigma, size=(m, m))
        self.V = np.random.normal(mu, sigma, size=(self.K, m))

        self.m_b = np.zeros((self.m, 1))
        self.m_c = np.zeros((self.K, 1))
        self.m_U = np.zeros((self.m, self.K))
        self.m_W = np.zeros((self.m, self.m))
        self.m_V = np.zeros((self.K, self.m))

    def forward_pass(self, h0, x):
        a = np.matmul(self.W,h0) + np.matmul(self.U,x) + self.b
        h1 = _tanh(a)
        o = np.matmul(self.V,h1) + self.c
        p = _softmax(o)
        return a, h1, o, p

    def propogate_forward(self, x, y, hprev):
        n = len(x)
        loss = 0
        a = np.zeros((n , len(self.b), 1))
        h = np.zeros((n , len(self.b), 1))
        o = np.zeros((n , len(self.c), 1))
        p = np.zeros((n , len(self.c), 1))
        h[-1] = hprev
        
        # Forward pass
        for t in range(n):
            a[t], h[t], o[t], p[t] = self.forward_pass(h[t-1], x[t])
            loss += _cross_entropy(p[t], y[t])
        return a, h, o , p, loss

    def compute_gradients(self, x, y, hprev):
        n= len(x)

        #forward pass
        a, h, o, p, loss = self.propogate_forward(x, y, hprev) 

        H0 = np.zeros((n , len(self.b), 1))
        H0[1:] = h[:-1]

        #initialize
        self.grad_b = np.zeros((self.m, 1))
        self.grad_c = np.zeros((self.K, 1))
        self.grad_U = np.zeros((self.m, self.K))
        self.grad_W = np.zeros((self.m, self.m))
        self.grad_V = np.zeros((self.K, self.m))

        self.grad_o = np.zeros_like(o[0])
        self.grad_h = np.zeros_like(h[0])
        self.grad_h_next = np.zeros_like(h[0])
        self.grad_a = np.zeros_like(a[0])
        
        # calculate gradients
        for t in reversed(range(n)):
            self.grad_o = np.copy(p[t] - y[t])
            self.grad_V += np.outer(self.grad_o,h[t])
            self.grad_c += self.grad_o
            self.grad_h = np.matmul(self.V.T,self.grad_o) + np.matmul(self.W.T,self.grad_a)
            self.grad_a = np.multiply(self.grad_h, (1 - np.square(h[t])))
            self.grad_b += self.grad_a
            self.grad_U += np.outer(self.grad_a,x[t].T)
            self.grad_W += np.dot(self.grad_a,H0[t].T)
            
  
        # clipping
        self.grad_V = np.clip(self.grad_V, -5, 5)
        self.grad_W = np.clip(self.grad_W, -5, 5)
        self.grad_U = np.clip(self.grad_U, -5, 5)
        self.grad_c = np.clip(self.grad_c, -5, 5)
        self.grad_b = np.clip(self.grad_b, -5, 5)

        # Update the hidden state sequence
        hprev = h[-1]

        return [self.grad_W,self.grad_U, self.grad_V, self.grad_b, self.grad_c], loss, hprev

    def compute_gradients_num(self, x, y, hprev, h):
        self.grad_b_num = np.zeros((self.m, 1))
        self.grad_c_num = np.zeros((self.K, 1))
        self.grad_U_num = np.zeros((self.m, self.K))
        self.grad_W_num = np.zeros((self.m, self.m))
        self.grad_V_num = np.zeros((self.K, self.m))

        analytical_grads = {"W": self.W, "U": self.U, "V": self.V, "b": self.b, "c": self.c}
        numerical_grads  = {"W": self.grad_W_num, "U": self.grad_U_num, "V": self.grad_V_num, "b": self.grad_b_num, "c": self.grad_c_num}
        
        for key in analytical_grads:
            for i in range( analytical_grads[key].shape[0]):
                for j in range( analytical_grads[key].shape[1]):

                    analytical_grads[key][i,j] += h
                    _, loss_1, _ = self.compute_gradients(x, y, hprev)
                    analytical_grads[key][i,j] -= 2*h
                    _, loss_2, _ = self.compute_gradients(x, y, hprev)
                    analytical_grads[key][i,j] +=h
                    numerical_grads[key][i,j] = (loss_1 - loss_2) / (2*h)

        return list(numerical_grads.values())

    def synthesize_text(self, h, x, n):
            seq = ''
            for __ in range(n):
                _, h, _, p = self.forward_pass(h, x)
                i = np.random.choice(range(self.vocab_len), p=p.flat)
                x= get_one_hot_2(x,i,self.vocab_len)
                seq += self.ind_to_char[i]
            return seq

    def fit(self, num_epochs, eps):
        e = 0
        step = 0 
        first_iter = True
        n_seq = floor( (len(self.book_data)- (self.N +1)) / float(self.N))
        smooth_list = []
    
        for i in range(num_epochs):
            hprev = np.zeros((self.m, 1))
            e=0
            for seq in range(n_seq):
             
                inputs = [self.char_to_ind[char] for char in self.book_data[e:e+self.N]]
                targets = [self.char_to_ind[char] for char in self.book_data[e+1:e+self.N+1]]
                n = len(inputs)
                x = get_one_hot(inputs, self.vocab_len, n)
                y = get_one_hot(targets, self.vocab_len, n)
                
                _, loss, hprev = self.compute_gradients(x, y, hprev)
            
                if first_iter: 
                    smooth_loss = loss
                    first_iter = False
                smooth_loss = 0.999 * smooth_loss + 0.001 * loss
   
                # generate text
                if step % 10000 == 0:
                    txt = self.synthesize_text(hprev, x[0], 200)
                    print('Iteration' +str (step))
                    print(txt)
                    print('Smooth loss: '+ str(smooth_loss))
                    print('--'*50)

                # update weights
                self.m_b += np.multiply(self.grad_b, self.grad_b)
                self.m_c += np.multiply(self.grad_c, self.grad_c)
                self.m_U += np.multiply(self.grad_U, self.grad_U)
                self.m_W += np.multiply(self.grad_W, self.grad_W)
                self.m_V += np.multiply(self.grad_V, self.grad_V)

                self.b -= self.eta * np.multiply(1 / np.sqrt(self.m_b + eps), self.grad_b)
                self.c -= self.eta * np.multiply(1 / np.sqrt(self.m_c + eps), self.grad_c)
                self.U -= self.eta * np.multiply(1 / np.sqrt(self.m_U + eps), self.grad_U)
                self.W -= self.eta * np.multiply(1 / np.sqrt(self.m_W + eps), self.grad_W)
                self.V -= self.eta * np.multiply(1/ np.sqrt(self.m_V + eps), self.grad_V)

                e += self.N
                step += 1

                if smooth_loss < 41:
                    plot_performance( smooth_list)

                    txt = self.synthesize_text(hprev, x[0], 1000)        
                    
                    print('Text generated by best model: ')
                    print('Iteration: ' +str (step))
                    print()
                    print(txt)
                    print('Smooth loss: '+ str(smooth_loss))
                    print(DONE)

#-----------------------------------------------------------------------------
#MAIN
#-----------------------------------------------------------------------------

if __name__ == '__main__':
    task = 1

    #check gradient
    if task == 1:
        data = load_data(sys.argv[1])
        data_info = get_data_info(data)
        rnn = RNN(data_info)

        hprev = np.zeros((rnn.m, 1))
        e = 0
        inputs = [rnn.char_to_ind[char] for char in rnn.book_data[e:e+rnn.N]]
        targets = [rnn.char_to_ind[char] for char in rnn.book_data[e+1:e+rnn.N+1]]
        n = len(inputs)
        x = get_one_hot(inputs, rnn.vocab_len, n)
        y = get_one_hot(targets, rnn.vocab_len, n)

        grads_ana, _, _ = rnn.compute_gradients(x, y, hprev)
        grads_num = rnn.compute_gradients_num(x, y, hprev, 1e-4)
        
        check_gradients(grads_ana, grads_num)
    
    #train
    if task == 2:
        #settings
        num_epoch = 15
        eps = 1e-8

        data = load_data(sys.argv[1])
        data_info = get_data_info(data)
        rnn = RNN(data_info)
        rnn.fit(num_epoch,eps)
