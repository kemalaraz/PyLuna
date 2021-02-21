import numpy as np

from ..base import BaseLosses

class CrossEntropy(BaseLosses):

    @staticmethod
    def forward(predictions:np.ndarray, ground_truths:np.ndarray):
        return np.mean(np.multiply(ground_truths, np.log(predictions)) +
                        np.multiply((1-ground_truths), np.log(1-predictions)))

    @staticmethod
    def backward(*args, **kwargs):
        pass

class MSE(BaseLosses):

    @staticmethod
    def forward(predictions:np.ndarray, ground_truths:np.ndarray):
        return -0.5*np.mean(np.square(np.subtract(predictions, ground_truths)))

    @staticmethod
    def backward(inputs:np.ndarray, predictions:np.ndarray, ground_truths:np.ndarray):
        return np.mean(np.dot(np.subtract(predictions, ground_truths), inputs.T))