from keras.datasets import fashion_mnist, mnist
import numpy as np
import math
import wandb
import argparse
from types import SimpleNamespace

def process(x) :
  x_proc = x.reshape(len(x), -1)
  x_proc = x_proc.astype('float64')
  x_proc = x_proc / 255.0
  return x_proc

def load_data(dataset = "fashion_mnist"):
  if dataset == "fashion_mnist" :
      (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
  elif dataset == "mnist":
      (x_train, y_train), (x_test, y_test) = mnist.load_data()
  
  x_train, x_valid = x_train[:int(len(x_train) * 0.9)], x_train[int(len(x_train) * 0.9):]
  y_train, y_valid = y_train[:int(len(y_train) * 0.9)], y_train[int(len(y_train) * 0.9):]

  x_train = process(x_train)
  x_valid = process(x_valid)
  x_test = process(x_test) 

  k = 10
  y_train = np.eye(k)[y_train] # one-hot
  y_valid = np.eye(k)[y_valid]
  y_test = np.eye(k)[y_test]
  
  return x_train, y_train, x_valid, y_valid, x_test, y_test

def sigmoid(x) :
  return 1. / (1. + np.exp(-x))

def tanh(x) :
  return (2. / (1. + np.exp(-2.*x))) - 1.

def relu(x) : # do not use relu with random
  return np.where(x >= 0, x, 0.)

def softmax(x) :
  x = x - np.max(x, axis=0)
  y = np.exp(x)
  return y / y.sum(axis=0)

class my_nn :

  def __init__(self, n_feature = 784, n_class = 10, nhl = 1, sz = 4, weight_init = "random", act_fun = "sigmoid", loss = "cross_entropy", 
               epochs = 1, b_sz = 4, optimizer = "sgd", lr = 0.1, mom = 0.9, beta = 0.9, beta1 = 0.9, beta2 = 0.999, epsilon = 0.000001, w_d = 0.005) :
    self.n_feature = n_feature
    self.n_class = n_class
    self.nhl = nhl
    self.L = nhl + 1
    self.sz = sz
    self.weight_init = weight_init
    self.act_fun = act_fun
    self.loss = loss
    self.epochs = epochs
    self.b_sz = b_sz
    self.optimizer = optimizer
    self.lr = lr
    self.mom = mom
    self.beta = beta
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.w_d = w_d

    self.W = [0 for i in range(0, self.L+1, 1)]
    self.b = [0 for i in range(0, self.L+1, 1)]

    self.d_a = [0 for i in range(0, self.L+1, 1)]
    self.d_b = [0 for i in range(0, self.L+1, 1)]
    self.d_W = [0 for i in range(0, self.L+1, 1)]

    self.a = [0 for i in range(0, self.L+1, 1)]
    self.h = [0 for i in range(0, self.L+1, 1)]

    self.u_W = [0 for i in range(0, self.L+1, 1)]
    self.u_b = [0 for i in range(0, self.L+1, 1)]

    self.W_look = [0 for i in range(0, self.L+1, 1)]
    self.b_look = [0 for i in range(0, self.L+1, 1)]

    self.v_W = [0 for i in range(0, self.L+1, 1)]
    self.v_b = [0 for i in range(0, self.L+1, 1)]

    self.m_W = [0 for i in range(0, self.L+1, 1)]
    self.m_b = [0 for i in range(0, self.L+1, 1)]

    self.initialization()

  ######################################################

  def initialization(self) :
    if self.act_fun == "ReLU" :
      self.W[1] = np.random.randn(self.sz, self.n_feature) * np.sqrt(2.0/self.n_feature)
      for i in range(2, self.L, 1) :
        self.W[i] = np.random.randn(self.sz, self.sz) * math.sqrt(2.0/self.sz)
      self.W[self.L] = np.random.randn(self.n_class, self.sz) * math.sqrt(2.0/self.sz)

    elif self.weight_init == "random" :
      self.W[1] = np.random.randn(self.sz, self.n_feature)
      for i in range(2, self.L, 1) :
        self.W[i] = np.random.randn(self.sz, self.sz)
      self.W[self.L] = np.random.randn(self.n_class, self.sz)

    elif self.weight_init == "Xavier" :
      self.W[1] = np.random.randn(self.sz, self.n_feature) * np.sqrt(2.0/self.n_feature)
      for i in range(2, self.L, 1) :
        self.W[i] = np.random.randn(self.sz, self.sz) * math.sqrt(2.0/self.sz)
      self.W[self.L] = np.random.randn(self.n_class, self.sz) * math.sqrt(2.0/self.sz)
    
    for i in range(1, self.L, 1) :
      self.b[i] = np.zeros((self.sz, 1))
    self.b[self.L] = np.zeros((self.n_class, 1))
  
  #########################################################

  def forward_propagation(self, x) :
    self.h[0] = x

    for i in range(1, self.L, 1) :
      self.a[i] = self.b[i] + np.dot(self.W[i], self.h[i-1])

      if self.act_fun == "sigmoid" :
        self.h[i] = sigmoid(self.a[i])
      elif self.act_fun == "tanh" :
        self.h[i] = tanh(self.a[i])
      elif self.act_fun == "ReLU" :
        self.h[i] = relu(self.a[i])
    
    self.a[self.L] = self.b[self.L] + np.dot(self.W[self.L], self.h[self.L-1])
    self.h[self.L] = softmax(self.a[self.L]) # h[L] = y_hat

  #########################################################

  def back_propagation(self, y) :
    if self.loss == "cross_entropy" :
      self.d_a[self.L] = self.h[self.L] - y
    elif self.loss == "mean_squared_error" :
      self.d_a[self.L] = (self.h[self.L] - y) * (self.h[self.L] * (1. - self.h[self.L]))
    
    self.d_b[self.L] = np.sum(self.d_a[self.L], axis=1, keepdims=True)
    self.d_W[self.L] = np.dot(self.d_a[self.L], self.h[self.L-1].T) + self.w_d * self.W[self.L]
    
    for i in range(self.L-1, 0, -1) :
      d_h_i = np.dot(self.W[i+1].T, self.d_a[i+1])
      
      if self.act_fun == "sigmoid" :
        g_dash_a_i = self.h[i] * (1. - self.h[i])
      elif self.act_fun == "tanh" :
        g_dash_a_i = 1. - self.h[i]**2
      elif self.act_fun == "ReLU" :
        g_dash_a_i = np.where(self.h[i] > 0., 1., 0.)
      
      self.d_a[i] = d_h_i * g_dash_a_i
      self.d_b[i] = np.sum(self.d_a[i], axis=1, keepdims=True)
      self.d_W[i] = np.dot(self.d_a[i], self.h[i-1].T) + self.w_d * self.W[i]

  ############################################################

  def nag_forward_propagation(self, x) :
    self.h[0] = x

    for i in range(1, self.L, 1) :
      self.a[i] = self.b_look[i] + np.dot(self.W_look[i], self.h[i-1])

      if self.act_fun == "sigmoid" :
        self.h[i] = sigmoid(self.a[i])
      elif self.act_fun == "tanh" :
        self.h[i] = tanh(self.a[i])
      elif self.act_fun == "ReLU" :
        self.h[i] = relu(self.a[i])
    
    self.a[self.L] = self.b_look[self.L] + np.dot(self.W_look[self.L], self.h[self.L-1])
    self.h[self.L] = softmax(self.a[self.L]) # h[L] = y_hat

  #########################################################

  def nag_back_propagation(self, y) :
    if self.loss == "cross_entropy" :
      self.d_a[self.L] = self.h[self.L] - y
    elif self.loss == "mean_squared_error" :
      self.d_a[self.L] = (self.h[self.L] - y) * (self.h[self.L] * (1. - self.h[self.L]))
    
    self.d_b[self.L] = np.sum(self.d_a[self.L], axis=1, keepdims=True)
    self.d_W[self.L] = np.dot(self.d_a[self.L], self.h[self.L-1].T) + self.w_d * self.W_look[self.L]
    
    for i in range(self.L-1, 0, -1) :
      d_h_i = np.dot(self.W_look[i+1].T, self.d_a[i+1])
      
      if self.act_fun == "sigmoid" :
        g_dash_a_i = self.h[i] * (1. - self.h[i])
      elif self.act_fun == "tanh" :
        g_dash_a_i = 1. - self.h[i]**2
      elif self.act_fun == "ReLU" :
        g_dash_a_i = np.where(self.h[i] > 0., 1., 0.)
      
      self.d_a[i] = d_h_i * g_dash_a_i
      self.d_b[i] = np.sum(self.d_a[i], axis=1, keepdims=True)
      self.d_W[i] = np.dot(self.d_a[i], self.h[i-1].T) + self.w_d * self.W_look[i]

  ############################################################

  def predict_prob(self, x) :
    a_temp = [0 for i in range(0, self.L+1, 1)]
    h_temp = [0 for i in range(0, self.L+1, 1)]
    h_temp[0] = x

    for i in range(1, self.L, 1) :
      a_temp[i] = self.b[i] + np.dot(self.W[i], h_temp[i-1])

      if self.act_fun == "sigmoid" :
        h_temp[i] = sigmoid(a_temp[i])
      elif self.act_fun == "tanh" :
        h_temp[i] = tanh(a_temp[i])
      elif self.act_fun == "ReLU" :
        h_temp[i] = relu(a_temp[i])
    
    a_temp[self.L] = self.b[self.L] + np.dot(self.W[self.L], h_temp[self.L-1])
    h_temp[self.L] = softmax(a_temp[self.L]) # h[L] = y_hat

    return h_temp[self.L].T
  
  #############################################################

  def loss_val(self, y_hat, y) :
    loss_val = 0.0
    N = y.shape[0]

    if self.loss == "cross_entropy" :
      for i in range(0, N, 1) :
        temp_loss = math.log(y_hat[i][y[i].argmax()])
        loss_val += temp_loss
      
      loss_val *= (-1.0/N)
    
    elif self.loss == "mean_squared_error" :
      loss_val = np.sum((y - y_hat)**2) / N

    return loss_val

  ##############################################################

  def accuracy(self, y_hat, y) :
    N = y.shape[0]
    n_correct = 0

    for i in range(0, N, 1) :
      if y[i].argmax() == y_hat[i].argmax() :
        n_correct += 1
    
    return 100 * n_correct / N

  ###############################################################

  def sgd(self, X, y, X_valid, y_valid) :
    t = 0
    N = X.shape[0]

    while t < self.epochs :
      for j in range(0, N, self.b_sz) :
        r_idx = j + self.b_sz
        if (j + self.b_sz) > N :
          r_idx = N
        self.forward_propagation(X[j:r_idx].T)
        self.back_propagation(y[j:r_idx].T)
        
        for idx in range(1, self.L+1, 1) :
          self.W[idx] = self.W[idx] - (self.lr * self.d_W[idx])
          self.b[idx] = self.b[idx] - (self.lr * self.d_b[idx])
      
      y_hat = self.predict_prob(X.T)
      tr_loss = self.loss_val(y_hat, y)
      tr_acc = self.accuracy(y_hat, y)

      y_val_hat = self.predict_prob(X_valid.T)
      val_loss = self.loss_val(y_val_hat, y_valid)
      val_acc = self.accuracy(y_val_hat, y_valid)

      print(f"epoch {t + 1} : train_loss = {tr_loss:.2f} valid_loss = {val_loss:.2f}, train accuracy = {tr_acc:.2f} valid_accuracy = {val_acc:.2f}")
      # wandb.log({'tr_loss' : tr_loss, 'tr_accuracy' : tr_acc, 'val_loss' : val_loss, 'val_accuracy' : val_acc})

      t += 1

  #################################################################

  def mgd(self, X, y, X_valid, y_valid) :
    t = 0
    N = X.shape[0]
    n_step = 0

    while t < self.epochs :
      for j in range(0, N, self.b_sz) :
        n_step += 1
        r_idx = j + self.b_sz
        if (j + self.b_sz) > N :
          r_idx = N
        self.forward_propagation(X[j:r_idx].T)
        self.back_propagation(y[j:r_idx].T)

        for idx in range(1, self.L+1, 1) :
          if n_step == 1 :
            self.u_W[idx] = (self.lr * self.d_W[idx])
            self.u_b[idx] = (self.lr * self.d_b[idx])
          else :
            self.u_W[idx] = (self.mom * self.u_W[idx]) + (self.lr * self.d_W[idx])
            self.u_b[idx] = (self.mom * self.u_b[idx]) + (self.lr * self.d_b[idx])
          
          self.W[idx] = self.W[idx] - self.u_W[idx]
          self.b[idx] = self.b[idx] - self.u_b[idx]

      y_hat = self.predict_prob(X.T)
      tr_loss = self.loss_val(y_hat, y)
      tr_acc = self.accuracy(y_hat, y)

      y_val_hat = self.predict_prob(X_valid.T)
      val_loss = self.loss_val(y_val_hat, y_valid)
      val_acc = self.accuracy(y_val_hat, y_valid)

      print(f"epoch {t + 1} : train_loss = {tr_loss:.2f} valid_loss = {val_loss:.2f}, train accuracy = {tr_acc:.2f} valid_accuracy = {val_acc:.2f}")
      # wandb.log({'tr_loss' : tr_loss, 'tr_accuracy' : tr_acc, 'val_loss' : val_loss, 'val_accuracy' : val_acc})
      
      t += 1

  ##################################################################

  def nagd(self, X, y, X_valid, y_valid) :
    t = 0
    N = X.shape[0]
    n_step = 0

    while t < self.epochs :
      for j in range(0, N, self.b_sz) :
        n_step += 1
        r_idx = j + self.b_sz
        if (j + self.b_sz) > N :
          r_idx = N
        if n_step == 1 :
          self.forward_propagation(X[j:r_idx].T)
          self.back_propagation(y[j:r_idx].T)
        else :
          for idx in range(1, self.L+1, 1) :
            self.W_look[idx] = self.W[idx] - (self.mom * self.u_W[idx])
            self.b_look[idx] = self.b[idx] - (self.mom * self.u_b[idx])
          self.nag_forward_propagation(X[j:r_idx].T)
          self.nag_back_propagation(y[j:r_idx].T)

        for idx in range(1, self.L+1, 1) :
          if n_step == 1 :
            self.u_W[idx] = (self.lr * self.d_W[idx])
            self.u_b[idx] = (self.lr * self.d_b[idx])
          else :
            self.u_W[idx] = (self.mom * self.u_W[idx]) + (self.lr * self.d_W[idx])
            self.u_b[idx] = (self.mom * self.u_b[idx]) + (self.lr * self.d_b[idx])
          
          self.W[idx] = self.W[idx] - self.u_W[idx]
          self.b[idx] = self.b[idx] - self.u_b[idx]
        
      y_hat = self.predict_prob(X.T)
      tr_loss = self.loss_val(y_hat, y)
      tr_acc = self.accuracy(y_hat, y)

      y_val_hat = self.predict_prob(X_valid.T)
      val_loss = self.loss_val(y_val_hat, y_valid)
      val_acc = self.accuracy(y_val_hat, y_valid)

      print(f"epoch {t + 1} : train_loss = {tr_loss:.2f} valid_loss = {val_loss:.2f}, train accuracy = {tr_acc:.2f} valid_accuracy = {val_acc:.2f}")
      # wandb.log({'tr_loss' : tr_loss, 'tr_accuracy' : tr_acc, 'val_loss' : val_loss, 'val_accuracy' : val_acc})
      t += 1

  ##############################################################

  def rmsprop(self, X, y, X_valid, y_valid) :
    t = 0
    N = X.shape[0]
    n_step = 0

    while t < self.epochs :
      for j in range(0, N, self.b_sz) :
        n_step += 1
        r_idx = j + self.b_sz
        if (j + self.b_sz) > N :
          r_idx = N
        self.forward_propagation(X[j:r_idx].T)
        self.back_propagation(y[j:r_idx].T)

        for idx in range(1, self.L+1, 1) :
          if n_step == 1 :
            self.v_W[idx] = ((1. - self.beta) * (self.d_W[idx]**2))
            self.v_b[idx] = ((1. - self.beta) * (self.d_b[idx]**2))
          else :
            self.v_W[idx] = (self.beta * self.v_W[idx]) + ((1. - self.beta) * (self.d_W[idx]**2))
            self.v_b[idx] = (self.beta * self.v_b[idx]) + ((1. - self.beta) * (self.d_b[idx]**2))
          
          self.W[idx] = self.W[idx] - (self.lr / (np.sqrt(self.v_W[idx] + self.epsilon))) * self.d_W[idx]
          self.b[idx] = self.b[idx] - (self.lr / (np.sqrt(self.v_b[idx] + self.epsilon))) * self.d_b[idx]
        
      y_hat = self.predict_prob(X.T)
      tr_loss = self.loss_val(y_hat, y)
      tr_acc = self.accuracy(y_hat, y)

      y_val_hat = self.predict_prob(X_valid.T)
      val_loss = self.loss_val(y_val_hat, y_valid)
      val_acc = self.accuracy(y_val_hat, y_valid)

      print(f"epoch {t + 1} : train_loss = {tr_loss:.2f} valid_loss = {val_loss:.2f}, train accuracy = {tr_acc:.2f} valid_accuracy = {val_acc:.2f}")
      # wandb.log({'tr_loss' : tr_loss, 'tr_accuracy' : tr_acc, 'val_loss' : val_loss, 'val_accuracy' : val_acc})
      t += 1
  
  ##############################################################

  def adam(self, X, y, X_valid, y_valid) :
    t = 0
    N = X.shape[0]
    n_step = 0

    while t < self.epochs :
      for j in range(0, N, self.b_sz) :
        n_step += 1
        r_idx = j + self.b_sz
        if (j + self.b_sz) > N :
          r_idx = N
        self.forward_propagation(X[j:r_idx].T)
        self.back_propagation(y[j:r_idx].T)

        for idx in range(1, self.L+1, 1) :
          if n_step == 1 :
            self.m_W[idx] = ((1. - self.beta1) * self.d_W[idx])
            self.m_b[idx] = ((1. - self.beta1) * self.d_b[idx])

            self.v_W[idx] = ((1. - self.beta2) * (self.d_W[idx]**2))
            self.v_b[idx] = ((1. - self.beta2) * (self.d_b[idx]**2))
          else :
            self.m_W[idx] = (self.beta1 * self.m_W[idx]) + ((1. - self.beta1) * self.d_W[idx])
            self.m_b[idx] = (self.beta1 * self.m_b[idx]) + ((1. - self.beta1) * self.d_b[idx])

            self.v_W[idx] = (self.beta2 * self.v_W[idx]) + ((1. - self.beta2) * (self.d_W[idx]**2))
            self.v_b[idx] = (self.beta2 * self.v_b[idx]) + ((1. - self.beta2) * (self.d_b[idx]**2))
          
          self.W[idx] = self.W[idx] - (self.lr / (np.sqrt(self.v_W[idx] / (1. - self.beta2**n_step) + self.epsilon))) * (self.m_W[idx] / (1. - self.beta1**n_step))
          self.b[idx] = self.b[idx] - (self.lr / (np.sqrt(self.v_b[idx] / (1. - self.beta2**n_step) + self.epsilon))) * (self.m_b[idx] / (1. - self.beta1**n_step))
        
      y_hat = self.predict_prob(X.T)
      tr_loss = self.loss_val(y_hat, y)
      tr_acc = self.accuracy(y_hat, y)

      y_val_hat = self.predict_prob(X_valid.T)
      val_loss = self.loss_val(y_val_hat, y_valid)
      val_acc = self.accuracy(y_val_hat, y_valid)

      print(f"epoch {t + 1} : train_loss = {tr_loss:.2f} valid_loss = {val_loss:.2f}, train accuracy = {tr_acc:.2f} valid_accuracy = {val_acc:.2f}")
      # wandb.log({'tr_loss' : tr_loss, 'tr_accuracy' : tr_acc, 'val_loss' : val_loss, 'val_accuracy' : val_acc})
      t += 1

  ##############################################################

  def nadam(self, X, y, X_valid, y_valid) :
    t = 0
    N = X.shape[0]
    n_step = 0

    while t < self.epochs :
      for j in range(0, N, self.b_sz) :
        n_step += 1
        r_idx = j + self.b_sz
        if (j + self.b_sz) > N :
          r_idx = N
        self.forward_propagation(X[j:r_idx].T)
        self.back_propagation(y[j:r_idx].T)

        for idx in range(1, self.L+1, 1) :
          if n_step == 1 :
            self.m_W[idx] = ((1. - self.beta1) * self.d_W[idx])
            self.m_b[idx] = ((1. - self.beta1) * self.d_b[idx])

            self.v_W[idx] = ((1. - self.beta2) * (self.d_W[idx]**2))
            self.v_b[idx] = ((1. - self.beta2) * (self.d_b[idx]**2))
          else :
            self.m_W[idx] = (self.beta1 * self.m_W[idx]) + ((1. - self.beta1) * self.d_W[idx])
            self.m_b[idx] = (self.beta1 * self.m_b[idx]) + ((1. - self.beta1) * self.d_b[idx])

            self.v_W[idx] = (self.beta2 * self.v_W[idx]) + ((1. - self.beta2) * (self.d_W[idx]**2))
            self.v_b[idx] = (self.beta2 * self.v_b[idx]) + ((1. - self.beta2) * (self.d_b[idx]**2))
          
          W_term = (self.beta1 / (1. - self.beta1**n_step)) * self.m_W[idx]  + ((1. - self.beta1) / (1. - self.beta1**n_step)) * self.d_W[idx]
          b_term = (self.beta1 / (1. - self.beta1**n_step)) * self.m_b[idx]  + ((1. - self.beta1) / (1. - self.beta1**n_step)) * self.d_b[idx]

          self.W[idx] = self.W[idx] - (self.lr / (np.sqrt(self.v_W[idx] / (1. - self.beta2**n_step) + self.epsilon))) * W_term
          self.b[idx] = self.b[idx] - (self.lr / (np.sqrt(self.v_b[idx] / (1. - self.beta2**n_step) + self.epsilon))) * b_term
        
      y_hat = self.predict_prob(X.T)
      tr_loss = self.loss_val(y_hat, y)
      tr_acc = self.accuracy(y_hat, y)

      y_val_hat = self.predict_prob(X_valid.T)
      val_loss = self.loss_val(y_val_hat, y_valid)
      val_acc = self.accuracy(y_val_hat, y_valid)

      print(f"epoch {t + 1} : train_loss = {tr_loss:.2f} valid_loss = {val_loss:.2f}, train accuracy = {tr_acc:.2f} valid_accuracy = {val_acc:.2f}")
      # wandb.log({'tr_loss' : tr_loss, 'tr_accuracy' : tr_acc, 'val_loss' : val_loss, 'val_accuracy' : val_acc})
      t += 1

  ##############################################################

  def train(self, X_train, y_train, X_valid, y_valid) :
    if self.optimizer == "sgd" :
      self.sgd(X_train, y_train, X_valid, y_valid)
    elif self.optimizer == "momentum" :
      self.mgd(X_train, y_train, X_valid, y_valid)
    elif self.optimizer == "nag" :
      self.nagd(X_train, y_train, X_valid, y_valid)
    elif self.optimizer == "rmsprop" :
      self.rmsprop(X_train, y_train, X_valid, y_valid)
    elif self.optimizer == "adam" :
      self.adam(X_train, y_train, X_valid, y_valid)
    elif self.optimizer == "nadam" :
      self.nadam(X_train, y_train, X_valid, y_valid)

# change the default hyperparameters to the best
if __name__=="__main__":
  parser = argparse.ArgumentParser(description = 'Input Hyperparameters')
  parser.add_argument('-wp'   , '--wandb_project'  , type = str  , default = 'CS22M059', metavar = '')
  parser.add_argument('-we'   , '--wandb_entity'   , type = str  , default = 'CS22M059', metavar = '')
  parser.add_argument('-d'    , '--dataset'        , type = str  , default = 'fashion_mnist', metavar = '', choices = ["mnist", "fashion_mnist"])
  parser.add_argument('-e'    , '--epochs'         , type = int  , default = 10, metavar = '')
  parser.add_argument('-b'    , '--batch_size'     , type = int  , default = 16, metavar = '')
  parser.add_argument('-l'    , '--loss'           , type = str  , default = 'cross_entropy', metavar = '', choices = ["mean_squared_error", "cross_entropy"])
  parser.add_argument('-o'    , '--optimizer'      , type = str  , default = 'nadam', metavar = '', choices = ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
  parser.add_argument('-lr'   , '--learning_rate'  , type = float, default = 0.001, metavar = '')
  parser.add_argument('-m'    , '--momentum'       , type = float, default = 0.9, metavar = '')
  parser.add_argument('-beta' , '--beta'           , type = float, default = 0.9, metavar = '')
  parser.add_argument('-beta1', '--beta1'          , type = float, default = 0.9, metavar = '')
  parser.add_argument('-beta2', '--beta2'          , type = float, default = 0.999, metavar = '')
  parser.add_argument('-eps'  , '--epsilon'        , type = float, default = 1e-5, metavar = '')
  parser.add_argument('-w_d'  , '--weight_decay'   , type = float, default = 0, metavar = '')
  parser.add_argument('-w_i'  , '--weight_init'    , type = str  , default = 'Xavier', metavar = '', choices = ["random", "Xavier"])
  parser.add_argument('-nhl'  , '--num_layers'     , type = int  , default = 3, metavar = '')
  parser.add_argument('-sz'   , '--hidden_size'    , type = int  , default = 64, metavar = '')
  parser.add_argument('-a'    , '--activation'     , type = str  , default = 'tanh', metavar = '', choices = ["sigmoid", "tanh", "ReLU"])
  
  params = vars(parser.parse_args())
  # wandb.init(project = params['wandb_project'], config = params)
  print("Provided hyperparameters = ", params)
  print("Building the model...")
  params = SimpleNamespace(**params)
  x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(params.dataset)

  epochs = params.epochs
  nhl = params.num_layers
  sz = params.hidden_size
  w_d = params.weight_decay
  lr = params.learning_rate
  optimizer = params.optimizer
  b_sz = params.batch_size
  weight_init = params.weight_init
  act_fun = params.activation
  loss = params.loss
  mom = params.momentum
  beta = params.beta
  beta1 = params.beta1
  beta2 = params.beta2
  epsilon = params.epsilon

  nn_model = my_nn(epochs = epochs, nhl = nhl, sz = sz, w_d = w_d, lr = lr, optimizer = optimizer, b_sz = b_sz, weight_init = weight_init, act_fun = act_fun, 
                  loss = loss, mom = mom, beta = beta, beta1 = beta1, beta2 = beta2, epsilon = epsilon)
  nn_model.train(x_train, y_train, x_valid, y_valid)
  print("Model built successfully.")