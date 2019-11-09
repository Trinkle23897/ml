import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes=W.shape[1]
  num_train=X.shape[0]
  for i in xrange(num_train):
    scores=X[i].dot(W)
    scores-=np.max(scores)
    sum=np.exp(scores).sum()
    loss+=np.log(sum)-scores[y[i]]
    for j in xrange(num_classes):
      dW[:,j]+=np.exp(scores[j])/sum*X[i]
    dW[:,y[i]]-=X[i]
  loss=loss/num_train+reg*np.sum(W*W)
  dW=dW/num_train+2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores=X.dot(W)
  N,C=scores.shape
  scores-=np.max(scores,axis=1).reshape((N,1)).dot(np.ones((1,C)))
  sum=np.sum(np.exp(scores),axis=1)
  loss=np.sum(np.log(sum)-scores[np.arange(N),y])/N+reg*np.sum(W*W)
  cnt=np.zeros_like(scores)
  cnt+=np.exp(scores)/(sum.reshape((N,1)).dot(np.ones((1,C))))
  cnt[np.arange(N),y]-=1
  dW=X.T.dot(cnt)/N+2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

