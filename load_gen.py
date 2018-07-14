import scipy.misc
import numpy as np


class gen():

    def __init__(self):
        self.path = r'G:/wuwenda/alltest/'
        self.num = 0

    def compare(self, a):
        return (a[0:6], eval(a[10:-4]))

    def imread(self, path):
        im = scipy.misc.imread(path, mode='RGB').astype(np.float)
        im = im[:, :, 2] / 255 - im[:, :, 0] / 255 * 2 + 1
        im = np.reshape(im, (1, 100, 100, 1))
        return im

    def my_gen(self, a, b):
        while 1:
            pic = self.imread(self.path + a[self.num])
            lab = np.reshape(b[self.num], (1, 5))
            self.num += 1
            yield pic, lab
