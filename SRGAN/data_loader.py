import tensorflow as tf
import os
import glob
import cv2
import numpy as np

class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, dataset_dir, batch_size=16, img_res=(256, 256)):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.img_res = img_res
        self.image_paths = glob.glob(os.path.join(dataset_dir, '*'))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_image_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        imgs_hr, imgs_lr = self.__load_batch(batch_image_paths)
        return imgs_lr, imgs_hr

    def on_epoch_end(self):
        np.random.shuffle(self.image_paths)

    def __load_batch(self, batch_image_paths):
        imgs_hr = []
        imgs_lr = []
        for img_path in batch_image_paths:
            img = self.__load_image(img_path)
            imgs_hr.append(img)
            imgs_lr.append(self.__downscale_image(img))
        return np.array(imgs_lr), np.array(imgs_hr)

    def __load_image(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_res)
        img = img / 127.5 - 1.
        return img

    def __downscale_image(self, img):
        img_lr = cv2.resize(img, (self.img_res[0] // 4, self.img_res[1] // 4))
        return img_lr