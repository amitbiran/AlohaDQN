class StatesBuffer(object):
    def __init__(self,size):
        self.buff = []
        self._size = size

    def add(self,item):
        self.buff.insert(0,item)
        if(len(self.buff)>self._size):
            self.buff.pop(-1)
        a=0

    def reset(self):
        self.buff = []



# s =StatesBuffer(4)
# s.add(0)
# s.add(1)
# s.add(2)
# s.add(3)
# s.add(0)
# s.add(1)
# s.add(0)
# s.add(1)
# s.add(0)
# s.add(1)
# s.add(0)
# s.add(1)
# a=1