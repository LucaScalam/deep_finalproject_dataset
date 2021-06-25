from __future__ import print_function
from keras.callbacks import Callback

import cv2
import numpy as np
import os


class TrainCheck(Callback):
    def __init__(self, output_path, model_name):
        self.epoch = 0
        self.output_path = output_path
        self.model_name = model_name

    def result_map_to_img(self,res_map):
      img = np.zeros((256, 512, 3), dtype=np.uint8)
      res_map = np.squeeze(res_map)

      argmax_idx = np.argmax(res_map, axis=2)

      #
      person = (argmax_idx == 1)
      car = (argmax_idx == 2)
      road = (argmax_idx == 3)

      sidewalk = (argmax_idx == 4)
      vegetation = (argmax_idx == 5)
      building = (argmax_idx == 6)
      sky = (argmax_idx == 7)

      #Azul
      img[:, :, 0] = np.where(person, 60, 0) + \
      + np.where(car, 142, 0) + \
      np.where(road, 128, 0) + \
      np.where(sidewalk, 232, 0) + \
      np.where(vegetation, 35, 0) + \
      np.where(building, 70, 0) + \
      np.where(sky, 180, 0)
      #Verde
      img[:, :, 1] = np.where(person, 20, 0) + \
      np.where(car, 0, 0) + \
      np.where(road, 64, 0) + \
      np.where(sidewalk, 35, 0) + \
      np.where(vegetation, 142, 0) + \
      np.where(building, 70, 0) + \
      np.where(sky, 130, 0)
      #Rojo
      img[:, :, 2] = np.where(person, 220, 0) + \
      np.where(car, 0, 0) + \
      np.where(road, 128, 0) + \
      np.where(sidewalk, 244, 0) + \
      np.where(vegetation, 107, 0) + \
      np.where(building, 70, 0) + \
      np.where(sky, 70, 0)

      # img[:, :, 0] = np.where(sky, 50, 0)
      # img[:, :, 1] = np.where(sky, 50, 0)
      # img[:, :, 2] = np.where(sky, 50, 0)


      return img

    def get_result_map(b_size, y_img):
        y_img = np.squeeze(y_img, axis=3)
        result_map = np.zeros((b_size, 256, 512, 8))

        # labels = ['background', 
        # 'person', 
        # 'car','truck','bus','caravan','train',
        # 'road','sidewalk',
        # 'vegetation',
        # 'building','wall','fence','guard_rail','bridge','tunnel','sky']

        #
        truck = (y_img == 27)
        bus = (y_img == 28)
        caravan = (y_img == 29)
        train = (y_img == 31)
        sidewalk = (y_img == 8)
        vegetation = (y_img == 21)
        building = (y_img == 11)
        wall = (y_img == 12)
        fence = (y_img == 13)
        guard_rail = (y_img == 14)
        bridge = (y_img == 15)
        tunnel = (y_img == 16)
        sky = (y_img == 23)

        person = (y_img == 24)
        car = (y_img == 26)
        road = (y_img == 7)
        background = np.logical_not(person + car + road +
        truck + bus + caravan + train + sidewalk + vegetation + 
        building + wall + fence + guard_rail + bridge + tunnel +
        sky)



        return result_map


    def on_epoch_end(self, epoch, logs={}):
        self.epoch = epoch+1
        self.visualize('img/test.png')

    def visualize(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mean_train = np.load('mean_train.npy',allow_pickle=True)
        mean_train = np.reshape(mean_train,((256,512,3)))
        img = (img - mean_train)/255.
        print('visualizando')
        img = np.expand_dims(img, 0)

        pred = self.model.predict(img)
        res_img = self.result_map_to_img(pred[0])

        cv2.imwrite(os.path.join(self.output_path, self.model_name + '_epoch_' + str(self.epoch) + '.png'), res_img)
