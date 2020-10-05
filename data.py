from glob import glob
from sklearn.model_selection import train_test_split
import os
import cv2
import numpy as np
import tensorflow as tf
from configs import *
from typing import List
AUTOTUNE = tf.data.experimental.AUTOTUNE


def split_data(path=PATH, val_split=0.15, test_split=0.1):
    images = sorted(glob(os.path.join(path, 'images/*')))
    masks = sorted(glob(os.path.join(path, 'masks/*')))
    total_size = len(images)
    val_size = int(val_split * total_size)
    test_size = int(test_split * total_size)

    train_img, val_img = train_test_split(images, test_size=val_size, random_state=101)
    train_msk, val_msk = train_test_split(masks, test_size=val_size, random_state=101)

    train_img, test_img = train_test_split(train_img, test_size=test_size, random_state=101)
    train_msk, test_msk = train_test_split(train_msk, test_size=test_size, random_state=101)

    return (train_img, train_msk), (val_img, val_msk), (test_img, test_msk)


class data_loader(object):
    def __init__(self, image_paths: List[str], mask_paths: List[str],
        image_size: int=IMAGE_SIZE, augment: bool=True):

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.augment = augment

    def randomHueSaturationValue(self, image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
        if np.random.random() < u:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(image)
            hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
            hue_shift = np.uint8(hue_shift)
            h += hue_shift
            sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
            s = cv2.add(s, sat_shift)
            val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
            v = cv2.add(v, val_shift)
            image = cv2.merge((h, s, v))
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        return image


    def randomShiftScaleRotate(self, image, mask,
                               shift_limit=(-0.0, 0.0),
                               scale_limit=(-0.0, 0.0),
                               rotate_limit=(-0.0, 0.0),
                               aspect_limit=(-0.0, 0.0),
                               borderMode=cv2.BORDER_CONSTANT, u=0.5):
        if np.random.random() < u:
            height, width, channel = image.shape

            angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
            scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
            aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
            sx = scale * aspect / (aspect ** 0.5)
            sy = scale / (aspect ** 0.5)
            dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
            dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

            cc = np.math.cos(angle / 180 * np.math.pi) * sx
            ss = np.math.sin(angle / 180 * np.math.pi) * sy
            rotate_matrix = np.array([[cc, -ss], [ss, cc]])

            box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
            box1 = box0 - np.array([width / 2, height / 2])
            box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

            box0 = box0.astype(np.float32)
            box1 = box1.astype(np.float32)
            mat = cv2.getPerspectiveTransform(box0, box1)
            image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                        borderValue=(
                                            0, 0,
                                            0,))
            mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                       borderValue=(
                                           0, 0,
                                           0,))

        return image, mask


    def randomHorizontalFlip(self, image, mask, u=0.5):
        if np.random.random() < u:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        return image, mask


    def randomVerticleFlip(self, image, mask, u=0.5):
        if np.random.random() < u:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        return image, mask


    def randomRotate90(self, image, mask, u=0.5):
        if np.random.random() < u:
            image = np.rot90(image)
            mask = np.rot90(mask)

        return image, mask

    def resize(self, image, mask):
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        return image, mask

    def read(self, image_path, mask_path):
        image_path = image_path.decode()
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        mask_path = mask_path.decode()
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return image, mask

    def normalize(self, image, mask):
        image = image / 255.
        image = image * 3.2 - 1.6
        mask = mask / 255.
        mask[mask >= 0.5] = 1
        mask[mask <= 0.5] = 0
        mask = np.expand_dims(mask, axis=-1)
        return image, mask

    def map_fn(self, image_path, mask_path):

        def process(image_path, mask_path):
            image, mask = self.read(image_path, mask_path)
            image, mask = self.resize(image, mask)
            if self.augment:
                image = self.randomHueSaturationValue(image,
                                 hue_shift_limit=(-30, 30),
                                 sat_shift_limit=(-5, 5),
                                 val_shift_limit=(-15, 15))

                image, mask = self.randomShiftScaleRotate(image, mask,
                                     shift_limit=(-0.1, 0.1),
                                     scale_limit=(-0.1, 0.1),
                                     aspect_limit=(-0.1, 0.1),
                                     rotate_limit=(-0, 0))
                image, mask = self.randomHorizontalFlip(image, mask)
                image, mask = self.randomVerticleFlip(image, mask)
                image, mask = self.randomRotate90(image, mask)

            image, mask = self.normalize(image, mask)
            return image, mask
        
        image, mask = tf.numpy_function(process, [image_path, mask_path], [tf.float64, tf.float64])

        image.set_shape([self.image_size, self.image_size, 3])
        mask.set_shape([self.image_size, self.image_size, 1])
        return image, mask

    def tf_data(self, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((self.image_paths, self.mask_paths))
        dataset = dataset.shuffle(len(self.image_paths), seed=101)
        dataset = dataset.map(self.map_fn, num_parallel_calls=AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        return dataset
