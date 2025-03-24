class RunningMean(object):
    def __init__(self, batch_size=100):
        self.reset()
        self.batch_size = batch_size

    def reset(self):
        self.mean = 0
        self.size = 0

    def add(self, x):
        self.size = min(self.batch_size, (self.size +1))
        old_mean_weighted = (1 - (1 /self.size)) * self.mean
        new_value_weighted = (1 / self.size) * x
        self.mean = old_mean_weighted + new_value_weighted

    def update(self):
        return self.mean