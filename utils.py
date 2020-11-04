import numpy as np
from matplotlib import image as img
from imgaug import augmenters as iaa
import cv2
import random


DATA_DIR = './driving_log.csv'

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3   # 160 320 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """

    return image[60:-25, :, :]  # remove the sky and the car front


def resize(image):
    """
    Resize the image to the input shape used by the network model
    """

    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """

    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    """
    Combine all preprocess functions into one
    """

    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)

    return image


def load_image(image_path):
    """
    return the preprocessed image
    """

    image = img.imread(image_path)

    return image


def zoom(image):
    """
    applying zoom augmenter
    """

    zoom_aug = iaa.Affine(scale=(1, 1.3))
    image = zoom_aug.augment_image(image)

    return image


def pan(image):
    """
    applying pan augmenters
    """

    pan_aug = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
    image = pan_aug.augment_image(image)

    return image


def img_random_brightness(image):
    """
    applying random brightness as a form of augmentation
    """

    brightness = iaa.Multiply((0.2, 1.2))
    image = brightness.augment_image(image)

    return image


def img_random_flip(image, steering_angle):
    """
    flipping images as a form of augmentation
    """

    image = cv2.flip(image, 1)
    steering_angle = -steering_angle
    return image, steering_angle


def random_augment(image, steering_angle):
    image = img.imread(image)
    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = img_random_brightness(image)
    if np.random.rand() < 0.5:
        image, steering_angle = img_random_flip(image, steering_angle)

    return image, steering_angle


def batch_generator(image_paths, steering_ang, batch_size, is_training):
    """
    feeding batches to our model
    """

    while True:

        batch_img = []
        batch_steering = []

        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths)-1)

            if is_training:
                image, steering = random_augment(image_paths[random_index, 0], steering_ang[random_index])

            else:
                image = load_image(image_paths[random_index, 0])
                steering = steering_ang[random_index]

            im = preprocess(image)
            batch_img.append(image)
            batch_steering.append(steering)

        yield (np.asarray(batch_img), np.asarray(batch_steering))