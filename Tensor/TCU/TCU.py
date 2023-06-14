from math import sqrt
from threading import Thread
from typing import List


from Tensor import DotProductUnit


from utils.args import args
from utils.formats import CustomData


class TCU(Thread, object):
    def __init__(self, tcu:int=0, **kwargs):
        Thread.__init__(self)
        total_dpu: int = args.tensor.tcu.dpu
        self.__rows, self.__columns = int(sqrt(total_dpu)), int(sqrt(total_dpu))
        self.__dpus: List[List[DotProductUnit]] = []
        self.__tcu = tcu
        self.__a: List[List[List[CustomData]]] = []
        self.__b: List[List[List[CustomData]]] = []
        self.__c: List[CustomData] = []
        self.__d: List[List[CustomData]] = []
        self.kwargs = kwargs
        self.__build()

    def __build(self):
        for row in range(self.__rows):
            row_dpu: List[DotProductUnit] = []
            for column in range(self.__columns):
                row_dpu.append(DotProductUnit(row, column, self.__tcu))
            self.__dpus.append(row_dpu)
    
    def run(self):
        self.mul()
    
    def mul(self):
        self.d = []
        for row, dpus_row in enumerate(self.__dpus):
            for column, dpu in enumerate(dpus_row):
                dpu.a = self.a[row][column]
                dpu.b = self.b[row][column]
                dpu.c = self.c[row][column]
                dpu.start()
        
        for dpus_row in self.__dpus:
            for dpu in dpus_row:
                dpu.join()
        
        for dpus_row in self.__dpus:
            row_results: List[CustomData] = []
            for dpu in dpus_row:
                row_results.append(dpu.d)
            self.d.append(row_results)

    @property
    def tcu(self):
        return self.__tcu
    
    @tcu.setter
    def tcu(self, tcu_id:int=0):
        self.__tcu = tcu_id

    @property
    def a(self) -> List[List[List[CustomData]]]:
        return self.__a
    
    @a.setter
    def a(self, a:List[List[List[CustomData]]]):
        assert len(a) == 4, ValueError
        self.__a = a

    @property
    def b(self) -> List[List[List[CustomData]]]:
        return self.__b
    
    @b.setter
    def b(self, b:List[List[List[CustomData]]]):
        assert len(b) == 4, ValueError
        self.__b = b

    @property
    def c(self) -> List[List[CustomData]]:
        return self.__c
    
    @c.setter
    def c(self, c:List[List[CustomData]]):
        self.__c = c
    
    @property
    def d(self) -> List[List[CustomData]]:
        return self.__d
    
    @d.setter
    def d(self, d:List[List[CustomData]]):
        self.__d = d
