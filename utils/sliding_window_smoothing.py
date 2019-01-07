from collections import OrderedDict

class AverageWithinWindow():
    def __init__(self, win_size):
        self.win_size = win_size
        self.cache = []
        self.average = 0
        self.count = 0

    def update(self, v):
        if self.count < self.win_size:
            self.cache.append(v)
            self.count += 1
            self.average = (self.average * (self.count - 1) + v) / self.count
        else:
            idx = self.count % self.win_size
            self.average += (v - self.cache[idx]) / self.win_size
            self.cache[idx] = v
            self.count += 1


class DictAccumulator():
    def __init__(self, win_size=None):
        self.accumulator = OrderedDict()
        self.total_num = 0
        self.win_size = win_size

    def update(self, d):
        self.total_num += 1
        for k, v in d.items(): ## AttributeError: 'Tensor' object has no attribute 'items'
            if not self.win_size:
                self.accumulator[k] = v + self.accumulator.get(k,0)
            else:
                self.accumulator.setdefault(k, AverageWithinWindow(self.win_size)).update(v)

    def get_average(self):
        average = OrderedDict()
        for k, v in self.accumulator.items():
            if not self.win_size:
                average[k] = v*1.0/self.total_num
            else:
                average[k] = v.average
        return average