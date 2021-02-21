import numpy as np
from typing import List

class BaseExceptions(BaseException):

    @abstractmethod
    def type_exceptions(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def shape_exceptions(self, *args, **kwargs):
        raise NotImplementedError