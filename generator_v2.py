import h5py
import numpy as np
import random
import cv2
from keras.preprocessing.image import ImageDataGenerator

mean_train = np.load('mean_train.npy',allow_pickle=True)
mean_train = np.reshape(mean_train,((256,512,3)))

labels = ['background', 
'person', 
'car','truck','bus','caravan','train',
'road','sidewalk',
'vegetation',
'building','wall','fence','guard rail','bridge','tunnel','sky']

def pre_processing(img):
    global mean_train
    return (img - mean_train)/255.


# Get ImageDataGenerator arguments(options) depends on mode - (train, val, test)
def get_data_gen_args(mode):
    if mode == 'train' :
        x_data_gen_args = dict(preprocessing_function=pre_processing,
                               shear_range=0.1,
                               zoom_range=0.1,
                               rotation_range=10,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               fill_mode='constant',
                               horizontal_flip=True)

        y_data_gen_args = dict(shear_range=0.1,
                               zoom_range=0.1,
                               rotation_range=10,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               fill_mode='constant',
                               horizontal_flip=True)

    elif (mode == 'test' or mode == 'val'):
        x_data_gen_args = dict(preprocessing_function=pre_processing)
        y_data_gen_args = dict()

    else:
        print("Parametro erróneo")
        return -1

    return x_data_gen_args, y_data_gen_args


# One hot encoding
def get_result_map(b_size, y_img):
    y_img = np.squeeze(y_img, axis=3)
    result_map = np.zeros((b_size, 256, 512, 8))

    # labels = ['background', 
    # 'person', 
    # 'car','truck','bus','caravan','train',
    # 'road','sidewalk',
    # 'vegetation',
    # 'building','wall','fence','guard_rail','bridge','tunnel','sky']

    # For np.where calculation.
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

    result_map[:, :, :, 0] = np.where(background, 1, 0)
    result_map[:, :, :, 1] = np.where(person, 1, 0)
    result_map[:, :, :, 2] = np.where(car + truck +
    bus + caravan + train , 1, 0)
    result_map[:, :, :, 3] = np.where(road, 1, 0)
    
    result_map[:, :, :, 4] = np.where(sidewalk, 1, 0)
    result_map[:, :, :, 5] = np.where(vegetation, 1, 0)
    result_map[:, :, :, 6] = np.where(building + wall + fence
    + guard_rail + bridge + tunnel, 1, 0)
    result_map[:, :, :, 7] = np.where(sky, 1, 0)


    return result_map


# para fit_generator.
def data_generator(d_path, b_size, mode):
    data = h5py.File(d_path, 'r')
    x_imgs = data.get('/' + mode + '/x')
    y_imgs = data.get('/' + mode + '/y')


    x_data_gen_args, y_data_gen_args = get_data_gen_args(mode)
    x_data_gen = ImageDataGenerator(**x_data_gen_args)
    y_data_gen = ImageDataGenerator(**y_data_gen_args)

    # random index
    d_size = x_imgs.shape[0]
    shuffled_idx = list(range(d_size))

    x = []
    y = []
    while True:
        random.shuffle(shuffled_idx)
        for i in range(d_size):
            idx = shuffled_idx[i]

            x.append(x_imgs[idx].reshape((256, 512, 3)))
            y.append(y_imgs[idx].reshape((256, 512, 1)))

            if len(x) == b_size:
                #data augmentation.
                _ = np.zeros(b_size)
                seed = random.randrange(1, 1000)

                x_tmp_gen = x_data_gen.flow(np.array(x), _,
                                            batch_size=b_size,
                                            seed=seed)
                y_tmp_gen = y_data_gen.flow(np.array(y), _,
                                            batch_size=b_size,
                                            seed=seed)

                
                x_result, _ = next(x_tmp_gen)
                y_result, _ = next(y_tmp_gen)

                yield x_result, get_result_map(b_size, y_result)

                x.clear()
                y.clear()
