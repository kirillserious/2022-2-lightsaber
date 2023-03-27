from typing import List, Optional

class Model:
    '''Содержит параметры модели'''
    def __init__(self, l:List[float], m:List[float], I:Optional[List[float]]=None, g:Optional[float]=9.8):
        if len(l) != len(m):
            raise Exception('Incorrect input shapes')
        if I is not None and len(I) != len(l):
            raise Exception('Incorrect input shapes')
        
        if I is None:
            I = [m[i] * l[i] * l[i] / 3 for i in range(len(l))]
        
        self.l = l
        self.m = m
        self.I = I
        self.g = g

    def dim(self)->int:
        return len(self.l)
