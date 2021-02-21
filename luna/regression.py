import numpy as np

from .._base import BaseModel
from ..utils.loss import MSE

class LinearRegression(BaseModel):

    __losses__ = {
                     "mean_squared_error": MSE
                 }

    @staticmethod
    def train(*args, **kwargs):
        return super().train(*args, **kwargs)

    @staticmethod
    def predict(*args, **kwargs):
        return super().predict(*args, **kwargs)