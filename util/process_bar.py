import time
import math


class ProcessBar:
    def __init__(self, _all, symbol='>'):
        self.start_ts = time.time()

        self.symbol = symbol
        self.all = _all
        self.last_ts = self.start_ts

    def update(self, now):
        ts2 = time.time()
        ratio = now/self.all * 100
        print(f'{self.symbol * int(ratio)} 进度:{now}/{self.all}({round(ratio, 2)}%) \
用时:{int(ts2-self.last_ts)}_{int(ts2 - self.start_ts)}/{math.ceil((ts2 - self.start_ts) / now * self.all)}  \r', end='')
        self.last_ts = ts2

    def end(self, end_str=''):
        print('')
        if end_str:
            print(end_str)
