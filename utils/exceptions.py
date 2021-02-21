import numpy as np
from typing import List

from .._base_exceptions import BaseExceptions

class Exceptions(BaseExceptions):

    def type_exceptions(self, variable, type):
        assert isinstance(variable, type), f"Predictions must be type numpy however got type -> {type(variable)}"

    def shape_exceptions(self, variable1, variable2):
        assert variable1.shape[0] == variable2.shape[0], "Ground truths and predictions have different number of instances"