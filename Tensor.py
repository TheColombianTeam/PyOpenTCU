import numpy as np


from FaultInjector import FaultInjector


class Tensor(FaultInjector):
    def __init__(self,
                threads_per_warp=32,
                tensor_buffer=4
                ) -> None:
        super.__init__(self)
        self._threads_per_warp = threads_per_warp
        self._tensor_buffer = tensor_buffer

    def mul(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:

        return np.zeros(0, 0)

