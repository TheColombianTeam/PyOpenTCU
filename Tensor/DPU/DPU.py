from typing import List
from threading import Thread


from utils.formats import CustomData


class DotProductUnit(Thread, object):
    def __init__(self, row:int=0, column:int=0, tcu:int=0, **kwargs):
        Thread.__init__(self)
        self.__row = row
        self.__column = column
        self.__tcu = tcu
        self.__a: List[CustomData] = []
        self.__b: List[CustomData] = []
        self.__c: CustomData = CustomData(0.0)
        self.__d: CustomData = CustomData(0.0)
        self.kwargs = kwargs

    def run(self):
        self.mul()

    def mul(self) -> CustomData:
        dot_product = CustomData(0.0)
        for index in range(4):
            dot_product += CustomData(self.__a[index]) * CustomData(self.__b[index])
        self.__d = dot_product + CustomData(self.__c)

    @property
    def row(self) -> int:
        return self.__row
    
    @row.setter
    def row(self, row_id:int=0):
        self.__row = row_id
    
    @property
    def column(self) -> int:
        return self.__column
    
    @column.setter
    def column(self, column_id:int=0):
        self.__column = column_id

    @property
    def tcu(self) -> int:
        return self.__tcu
    
    @tcu.setter
    def tcu(self, tcu_id:int=0):
        self.__tcu = tcu_id

    @property
    def a(self) -> List[CustomData]:
        return self.__a
    
    @a.setter
    def a(self, a:List[CustomData]):
        assert len(a) == 4, ValueError
        self.__a = a

    @property
    def b(self) -> List[CustomData]:
        return self.__b
    
    @b.setter
    def b(self, b:List[CustomData]):
        assert len(b) == 4, ValueError
        self.__b = b

    @property
    def c(self) -> CustomData:
        return self.__c
    
    @c.setter
    def c(self, c:CustomData):
        self.__c = c
    
    @property
    def d(self) -> CustomData:
        return self.__d
    
    @d.setter
    def d(self, d:CustomData):
        self.__d = d
