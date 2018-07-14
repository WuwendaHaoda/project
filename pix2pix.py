from __future__ import print_function, division
from keras.layers import Input, Dropout, Concatenate
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, ZeroPadding2D
from keras.models import Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
from data_loader import DataLoader
import numpy as np
import os
from keras.models import load_model

from openpyxl import load_workbook

from email import encoders
from email.header import Header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.utils import parseaddr, formataddr

import smtplib
from_addr = 'm15692219007@163.com'
password = 'xxxxxx'
to_addr = '841889390@qq.com'
smtp_server = 'smtp.163.com'


def _format_addr(s):
    name, addr = parseaddr(s)
    return formataddr((Header(name, 'utf-8').encode(), addr))


def sendtome(epo, dl1, dl2, gl1, gl2, gl3):
    ms = MIMEMultipart()
    ms['From'] = _format_addr('实验室 <%s>' % from_addr)
    ms['To'] = _format_addr('管理员 <%s>' % to_addr)
    ms['Subject'] = Header('%d效果' % epo, 'utf-8').encode()

    ms.attach(MIMEText('dloss:%.6f,%.6f\ngloss:%.6f,%.6f,%.6f' %
                       (dl1, dl2, gl1, gl2, gl3), 'plain', 'utf-8'))
    with open(r'G:\wuwenda\pixgen\%d\gen.png' % epo, 'rb') as f:
        mime = MIMEBase('image', 'png', filename='gen.png')
        mime.add_header('Content-Disposition',
                        'attachment', filename='gen.png')
        mime.add_header('Content-ID', '<0>')
        mime.add_header('X-Attachment-Id', '0')
        mime.set_payload(f.read())
        encoders.encode_base64(mime)
        ms.attach(mime)

    server = smtplib.SMTP_SSL(smtp_server, 465)  # SMTP协议默认端口是25
    server.login(from_addr, password)
    server.sendmail(from_addr, [to_addr], ms.as_string())
    server.quit()


def compare(a):
    return (a[0:6], eval(a[10:-4]))

for root, dirs, files in os.walk('G:/wuwenda/windpictures/'):
    files = sorted(files, key=compare)

x_tr = files[:-5]
y_tr = files[5:]

name_len = len(x_tr)

for i in range(name_len - 1, -1, -1):
    if x_tr[i][:6] != y_tr[i][:6]:
        x_tr.pop(i)
        y_tr.pop(i)

for root, dirs, files in os.walk('G:/wuwenda/testsample/'):
    files = sorted(files, key=compare)
x_te = files[::2]
y_te = files[1::2]

testnum = 4


class Pix2Pix():

    def __init__(self):
        # Input shape
        self.img_rows = 100
        self.img_cols = 100
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.dataset_name = 'windpictures'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))

        # Calculate output shape of D (PatchGAN)
        # patch = int(self.img_rows / 2**4)
        self.disc_patch = (7, 7, 1)

        # Number of filters in the first layer of G and D（第一层）
        self.gf = 16  # 原来是64
        self.df = 16

        self._build_model()

    def _build_model(self):
        optimizer = Adam(0.005, 0.5)
        optimizer1 = Adam(0.00001, 0.5)

        # Build and compile the discriminator
        self.discriminator = self._build_discriminator()
        self.discriminator.compile(loss='mse',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self._build_generator()
        self.generator.compile(
            loss='binary_crossentropy', optimizer=optimizer1)

        # Input images and their conditioning images
        before = Input(shape=self.img_shape)  # 后5天
        # img_A_add = ZeroPadding2D(padding=(14, 14))(img_A)
        after = Input(shape=self.img_shape)  # 前5天
        # img_B_add = ZeroPadding2D(padding=(14, 14))(img_B)

        # By conditioning on B generate a fake version of A
        fake_after = self.generator(before)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition
        # pairs
        valid = self.discriminator([before, fake_after])

        self.combined = Model([before, after], [fake_after, valid])
        self.combined.compile(loss=['mae', 'mse'],
                              loss_weights=[75, 25],
                              optimizer=optimizer1)

    def _build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size,
                       strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, need_reshape=False, f_size=4, dropout_rate=0.5):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)

            if not need_reshape:
                u = Conv2D(filters, kernel_size=f_size, strides=1,
                           padding='same', activation='relu')(u)
            else:
                u = Conv2D(filters, kernel_size=2, strides=1,
                           padding='valid', activation='relu')(u)

            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)
        # d_add = ZeroPadding2D(padding=(14, 14))(d0)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)  # 50
        d2 = conv2d(d1, self.gf * 2)  # 25
        d3 = conv2d(d2, self.gf * 4)  # 13
        d4 = conv2d(d3, self.gf * 8)  # 7
        d5 = conv2d(d4, self.gf * 8)  # 4
        d6 = conv2d(d5, self.gf * 8)  # 2

        # Upsampling
        u1 = deconv2d(d6, d5, self.gf * 8)  # 4
        u2 = deconv2d(u1, d4, self.gf * 8, True)  # 7
        u3 = deconv2d(u2, d3, self.gf * 4, True)  # 13
        u4 = deconv2d(u3, d2, self.gf * 2, True)  # 25
        u5 = deconv2d(u4, d1, self.gf)  # 50

        u6 = UpSampling2D(size=2)(u5)
        output_img = Conv2D(self.channels, kernel_size=4,
                            strides=1, padding='same', activation='tanh')(u6)  # 100

        return Model(d0, output_img)

    def _build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size,
                       strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])
        # combined_imgs = ZeroPadding2D(padding=(14, 14))(combined_imgs)

        d1 = d_layer(combined_imgs, self.df, bn=False)  # 50
        d2 = d_layer(d1, self.df * 2)  # 25
        d3 = d_layer(d2, self.df * 4)  # 13
        d4 = d_layer(d3, self.df * 8)  # 7

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=1, save_interval=50):

        start_time = datetime.datetime.now()  # 记录开始训练时间

        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):

            # ----------------------
            #  Train Discriminator
            # ----------------------
            before, after = self.data_loader.load_data(
                batch_size, x=x_tr, y=y_tr)  # 把两个列表的图片读入

            fake_after = self.generator.predict(
                before)  # 所以imgs_B是前5天，fake_A是后5天的预测

            d_loss_real = self.discriminator.train_on_batch(
                [before, after], valid)  # 明白了！对于每一对应该返回1(True)，所以这个是real_loss。[img_A,img_B](x)和valid(y)。
            d_loss_fake = self.discriminator.train_on_batch(
                [before, fake_after], fake)  # 同理↑
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ------------------
            #  Train Generator
            # ------------------
            g_loss = self.combined.train_on_batch(
                [before, after], [after, valid])

            elapsed_time = datetime.datetime.now() - start_time  # 记录一次epoch结束时间
            if (epoch + 1) % save_interval == 0:
                self.save_imgs(epoch, d_loss, g_loss)

    def save_imgs(self, epoch, a, b):
        os.makedirs('G:\wuwenda\pixgen\%d' % epoch, exist_ok=True)
        self.discriminator.save('G:\wuwenda\pixgen\%d\d.h5' % epoch)
        self.generator.save('G:\wuwenda\pixgen\%d\g.h5' % epoch)
        self.combined.save('G:\wuwenda\pixgen\%d\combined.h5' % epoch)

        before, after = self.data_loader.load_data(
            batch_size=testnum, is_testing=True, x=x_te, y=y_te)
        fake_after = self.generator.predict(before)

        titles = ['before', 'predict', 'after']
        fig, axs = plt.subplots(3, testnum)
        i = 0

        for it in [before, fake_after, after]:
            for ran in range(testnum):
                imgs_r = it[ran] * (-1)
                imgs_r[imgs_r < 0] = 0
                imgs_r = imgs_r / imgs_r.max()
                imgs_g = it[ran]
                imgs_g[imgs_g < 0] = 0
                imgs_g = imgs_g / imgs_g.max() / 2
                imgs_b = np.zeros((100, 100, 1)).astype(np.float)
                it_new = np.concatenate((imgs_r, imgs_g, imgs_b), axis=-1)
                new = np.array(it_new)
                axs[i, ran].imshow(new)
                axs[i, ran].set_title(titles[i])
                axs[i, ran].axis('off')
            i += 1

        # Rescale images 0 - 1
        # gen_imgs = 0.5 * gen_imgs + 0.5

        fig.savefig("G:\wuwenda\pixgen\%d\gen.png" % epoch)
        plt.close()

        wb = load_workbook(filename=r'G:\wuwenda\alllog.xlsx')
        ws = wb.active
        ws.append([epoch, a[0], a[1], b[0], b[1], b[2]])
        wb.save(filename=r'G:\wuwenda\alllog.xlsx')

        try:
            sendtome(epoch, a[0], a[1], b[0], b[1], b[2])
        except:
            print('sendfail')
        # quit()


if __name__ == '__main__':
    gan = Pix2Pix()  # 初始化模型
    gan.discriminator.load_weights(r'G:\wuwenda\pixgen4\29999\d.h5')
    gan.generator.load_weights(r'G:\wuwenda\pixgen4\29999\g.h5')
    # gan.combined = load_weights(r'G:\wuwenda\pixgen0\1600\combined.h5')
    gan.train(epochs=300000, batch_size=500, save_interval=500)  # 训练模型
