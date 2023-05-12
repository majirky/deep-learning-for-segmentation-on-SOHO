from PIL import Image, ImageDraw
# from ImageDataAugmentor.image_data_augmentor import *
import numpy as np
import sys

sys.path.append('../src/')

from model_scss_net import scss_net
from utils import create_contours
import os
import settings as Settings

IMG_SIZE = 256


def start_segmentation(path, event):
    """ for CH make segmentation with SCSS-Net model on CROPPED img and create contours of that segmentation on UNCROPPED EIT 195 image
    
        for AR make segmentation with SCSS-Net model on img and create contours of that segmentation on EIT 171 image

    Args:
        path (string): path to image to make prediction on (if event is CH path is to CROPPED 195 images, handled in webapp.py)

        event (string): "CH" or "AR"

    Returns:
        Image: image with countour of segmentation on provided image

        float: area coverage on Sun's disk of segmented event
    """

    if "default" in path:
        return Image.open(path), 0.0

    # select UNCROPPED 195A images or 171 images 
    if event == "CH":
        model_weights = "../modeling/ch_model.h5"
        img_src = Settings.IMAGES_195
        extention = ".jpg"
    else:
        model_weights = "../modeling/ar_model.h5"
        img_src = Settings.IMAGES_171
        extention = ".png"

    imgs_test = []
    imgs_test.append(path)

    imgs_test_list = []
    for image in imgs_test:
        imgs_test_list.append(np.array(Image.open(image).convert("L").resize((IMG_SIZE, IMG_SIZE))))


    x_test = np.asarray(imgs_test_list, dtype=np.float32)/255


    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    input_shape = x_test[0].shape

    # deep learning approach
    model = scss_net(
        input_shape,
        filters=32,
        layers=4,
        batch_norm=True,
        drop_prob=0.5)

    model.load_weights(model_weights)

    y_pred = model.predict(x_test)

    print(path)

    # make annotations on imgs
    annotations = create_contours(y_pred[0], target_size=(1024, 1024))
    image_name_clean = path[-29:-4]
    img = Image.open(img_src + image_name_clean + extention).resize((1024, 1024))
    draw = ImageDraw.Draw(img)

    for annotation in annotations:
        draw.polygon(annotation, outline="red", width=4)

    # calculating event coverage area in %
    size = 0
    for i in np.nditer(x_test):
        if i < 0.95:
            size = size + 1
    predicted_area = [(1 * (pixel > 0.1)).sum() for pixel in y_pred]

    if event == "CH":
        area_coverage = round((predicted_area[0]/size) * 100, 2)
        # note: 2012/01/26 - area coverage is 14.58%
    else:
        # use average size if cannot compute size of disk on image
        area_coverage = round((predicted_area[0]/28326) * 100, 2)

    return img, area_coverage