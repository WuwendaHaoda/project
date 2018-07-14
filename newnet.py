from data_loader import DataLoader
from load_gen import gen
import numpy as np
import pandas as pd
import os
import h5py

from openpyxl import load_workbook
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.header import Header
from email.utils import parseaddr, formataddr

from_addr = 'm15692219007@163.com'
password = 'xxxxxx'
to_addr = '841889390@qq.com'
smtp_server = 'smtp.163.com'


def _format_addr(s):
    name, addr = parseaddr(s)
    return formataddr((Header(name, 'utf-8').encode(), addr))


def sendtome(epo, a, b):
    msg = MIMEMultipart()
    msg['From'] = _format_addr('实验室 <%s>' % from_addr)
    msg['To'] = _format_addr('管理员 <%s>' % to_addr)
    msg['Subject'] = Header('%d效果' % epo, 'utf-8').encode()

    puretext = MIMEText('train_loss: %.6f,\ntest_loss: %.6f' %
                        (a[0], b[0]))
    msg.attach(puretext)

    xlsxpart = MIMEApplication(open(r'G:\wuwenda\alllog.xlsx', 'rb').read())
    xlsxpart.add_header('Content-Disposition',
                        'attachment', filename='alllog.xlsx')
    msg.attach(xlsxpart)
    server = smtplib.SMTP_SSL(smtp_server, 465)  # SMTP协议默认端口是25
    server.login(from_addr, password)
    server.sendmail(from_addr, [to_addr], msg.as_string())
    server.quit()


def compare(a):
    return (a[0:6], eval(a[10:-4]))
'''
for root, dirs, files in os.walk('G:/wuwenda/windpictures/'):
    files = sorted(files, key=compare)

x_tr = files[:-5]
y_tr = files[5:]
'''
d_tr = h5py.File(r'G:\wuwenda\labels_deleted.hdf5', 'r')['labels']
'''
name_len = len(x_tr)

for i in range(name_len - 1, -1, -1):
    if x_tr[i][:6] != y_tr[i][:6]:
        x_tr.pop(i)
        y_tr.pop(i)
        # d_tr = np.delete(d_tr, i, axis=0)
# print(len(x_tr))
# print(d_tr.shape)
# f = h5py.File(r'G:\wuwenda\labels_deleted.hdf5', 'w')
# d = f.create_dataset('labels', data=d_tr)
# f.close()

print(d_tr.shape)

pandas_file = pd.DataFrame(data=x_tr)  # 保存成csv
pandas_file.to_csv(r'G:\wuwenda\pictures_deleted.csv')
'''
with open(r'G:\wuwenda\pictures_deleted.csv') as f:
    x_tr = pd.read_csv(f).ix[:, 1]
# print(len(x_tr))

for root, dirs, files in os.walk('G:/wuwenda/alltest/'):
    files = sorted(files, key=compare)
d_te = h5py.File(r'G:\wuwenda\testlabels.hdf5', 'r')['labels']

import keras
from keras import backend as K
from keras import regularizers
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers.core import Flatten
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam


class net():

    def __init__(self, b_s):
        self.picture_shape = (100, 100, 1)
        self.data_loader = DataLoader()
        self.load_gen = gen()
        self.batch_size = b_s
        self._build_model()

    def _build_model(self):
        self.optimizer = Adam(lr=0.000001, beta_1=0.9,
                              beta_2=0.999, decay=0.00015)

        def my_accuracy3(y_true, y_pred):
            r = np.full((self.batch_size, 5), 0.45)
            r = K.variable(value=r)
            res = K.abs(y_true - y_pred)
            r = K.less(res, r)
            r = K.cast(r, dtype='float')
            return K.mean(r, axis=0, keepdims=True)

        def my_accuracy5(y_true, y_pred):
            r = np.full((self.batch_size, 5), 0.5)
            r = K.variable(value=r)
            res = K.abs(y_true - y_pred)
            r = K.less(res, r)
            r = K.cast(r, dtype='float')
            return K.mean(r, axis=0, keepdims=True)

        def allup_accuracy(y_true, y_pred):
            r = np.full((self.batch_size, 5), 0.5)
            r = K.variable(value=r)
            y_up = np.full((self.batch_size, 5), 1)
            y_up = K.variable(value=y_up)
            res = K.abs(y_true - y_up)
            r = K.less(res, r)
            r = K.cast(r, dtype='float')
            return K.mean(r, axis=0, keepdims=True)

        self.model = self._m()
        self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy',
                           metrics=[my_accuracy3, my_accuracy5, allup_accuracy])

    def _m(self):
        input_img = Input(self.picture_shape)
        x = keras.applications.xception.Xception(
            include_top=False, weights=None, input_tensor=input_img, input_shape=(100, 100, 1)).output
        '''
        def conv2d(layer, filters, size=3, stride=1, pad='valid', activation='relu', bn=True, dropout_rate=0.5):
            d = Conv2D(filters, kernel_size=size, strides=1, padding=pad, activation='relu',
                       kernel_regularizer=regularizers.l2(0.01))(layer)
            if dropout_rate:
                d = Dropout(dropout_rate)(d)
            if bn:
                d = BatchNormalization(momentum=0.99)(d)
            return d
        d0 = conv2d(input_img, 16, bn=False)  # 98
        d1 = conv2d(d0, 16)  # 96
        d2 = conv2d(d1, 16)  # 94
        d3 = conv2d(d2, 32)  # 92
        d4 = conv2d(d3, 32)  # 90
        d5 = conv2d(d4, 64, size=4, stride=2, pad='same')  # 45
        d6 = conv2d(d5, 64)  # 43
        d7 = conv2d(d6, 64)  # 41
        d8 = conv2d(d7, 128, size=4, stride=2, pad='same')  # 21
        d9 = conv2d(d8, 128)  # 19
        d10 = conv2d(d9, 128)  # 17
        d11 = conv2d(d10, 128)  # 15
        d12 = conv2d(d11, 256, size=4, stride=2, pad='same')  # 8
        d13 = conv2d(d12, 256)  # 6
        d14 = conv2d(d13, 256)  # 4
        d15 = conv2d(d14, 256)  # 2
        d16 = conv2d(d15, 512, size=2, stride=2, pad='same',)  # 1
        '''
        x = keras.layers.pooling.GlobalAveragePooling2D()(x)
        x = Dense(5, activation='sigmoid')(x)
        return Model(input_img, x)  # 这是一个loss

    def train(self, epochs, save_interval=50):
        for epoch in range(epochs):
            before, after = self.data_loader.load_data(
                self.batch_size, x=x_tr, y=d_tr)
            train_loss = self.model.train_on_batch(before, after)
            if epoch % save_interval == 0:
                self.saveresult(epoch, train_loss)

    def saveresult(self, epoch, a):
        os.makedirs('G:\wuwenda\logs\%d' % epoch, exist_ok=True)
        self.model.save('G:\wuwenda\logs\%d\save.hdf5' % epoch)

        test_loss = self.model.evaluate_generator(
            self.load_gen.my_gen(files, d_te), 1, len(files))
        self.load_gen.num = 0

        wb = load_workbook(filename=r'G:\wuwenda\alllog.xlsx')
        ws = wb.active
        ws.append([epoch] + a + test_loss)
        wb.save(filename=r'G:\wuwenda\alllog.xlsx')
        try:
            sendtome(epoch, a, test_loss)
        except:
            print('sendfail')


if __name__ == '__main__':
    my_net = net(125)
    my_net.model.load_weights(r'G:\wuwenda\logs0\84000\save.hdf5')
    my_net.train(epochs=300000000, save_interval=1000)
