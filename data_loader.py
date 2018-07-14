import scipy.misc
import numpy as np
import random


class DataLoader():

    def __init__(self):
        self.path = r'G:/wuwenda/windpictures/'

    def compare(self, a):
        return (a[0:6], eval(a[10:-4]))

    def imread(self, path):
        im = scipy.misc.imread(path, mode='RGB').astype(np.float)
        im = im[:, :, 2] / 255 - im[:, :, 0] / 255 * 2 + 1
        im = np.reshape(im, (100, 100, 1))
        return im

    def load_data(self, batch_size, x=[], y=[]):
        l = list(range(len(x)))
        l = random.sample(l, batch_size)
        before = []
        after = []
        for num in l:
            img = self.imread(self.path + x[num])
            before.append(img)
            after.append(y[num])
        before = np.array(before)
        after = np.array(after)
        return before, after
