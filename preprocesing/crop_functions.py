import numpy as np
import ephem
from datetime import datetime
import pandas as pd
from scipy.interpolate import interp1d
from PIL import Image, ImageDraw
import glob
from tqdm.notebook import tqdm, trange


def crop_and_save(path, path_to_save, base_margin_top, base_margin_right, base_margin_bottom, base_margin_left):
    """this is main function for croping sun's disk on EIT 195A images. Size of sun's disk is different troughtout year,
    because distance between SOHO and sun is different all around year.

    Base margins are numbers of pixels from margin to the sun disk on first image of the year. Image in markdown in ipynb file.

    Firstly we map sun distance for every day of selected year using map() function.

    Secondly we iterate every image file in folder provided in path argument and excract date from filename using get_date().
    
    Thirdly we use get_mask_margin() and get margins on how big sun's disk is and how big our pieclise in white foreground should be.

    Lastly we use crop_sun() that crops sun and saves that cropped sun to file

    Args:
        path (string): path to folder where are images of sun for one whole year (or more years, it does not really matter)

        path_to_save (string): path to folder where save image with cropped sun's disk

        base_margin_top (int): base margin is number of pixels from top end of image to the top edge of sun disk on first image of the year

        base_margin_right (int): base margin is number of pixels from right end of image to the right edge of sun disk on first image of the year

        base_margin_bottom (int): base margin is number of pixels from top bottom of image to the bottom edge of sun disk on first image of the year

        base_margin_left (int): base margin is number of pixels from left end of image to the left edge of sun disk on first image of the year
    """

    df = map(path, base_margin_top, base_margin_right, base_margin_bottom, base_margin_left)

    imgs_paths = glob.glob(path)
    imgs_paths = sorted(imgs_paths)

    for original_img in tqdm(imgs_paths):
        date = get_date(original_img)

        mt, mr, mb, ml = get_mask_margin(df, date)

        crop_sun(original_img, path_to_save, mt, mr, mb, ml)


def get_mask_margin(dataframe, date):
    """get margins for pieslice in crop_sun() from dataframe, which comes from map()

    Args:
        dataframe (pandas.Dataframe): dataframe contains margins for each day in year

        date (string): date for which to find margins format: 2002/01/31

    Returns:
        int: margin_top, margin_right, margin_bottom, margin_left 
    """

    margin_top = 0.0
    margin_right = 0.0
    margin_bottom = 0.0
    margin_left = 0.0

    for i in range(dataframe.shape[0]):
        if dataframe.time[i] == date:

            margin_top = dataframe.mask_top[i]
            margin_right = dataframe.mask_right[i]
            margin_bottom = dataframe.mask_bottom[i]
            margin_left = dataframe.mask_left[i]

    margin_top = int(margin_top)
    margin_right = int(margin_right)
    margin_bottom = int(margin_bottom)
    margin_left = int(margin_left)

    return margin_top, margin_right, margin_bottom, margin_left


def map(path, base_margin_top, base_margin_right, base_margin_bottom, base_margin_left):
    """ size of sun's disk is different troughtout year, because distance between SOHO and sun is different all around year.
    We have to map those distances to margins for pieslice in crop_sun()

    Args:
        path (string): path to folder that contains uncropped images of sundisk

        base_margin_top (int): base margin is number of pixels from top end of image to the top edge of sun disk on first image of the year

        base_margin_right (int): base margin is number of pixels from right end of image to the right edge of sun disk on first image of the year

        base_margin_bottom (int): base margin is number of pixels from top bottom of image to the bottom edge of sun disk on first image of the year

        base_margin_left (int): base margin is number of pixels from left end of image to the left edge of sun disk on first image of the year

    Returns:
        pandas.Dataframe: dataframe, that contains info about mapped margins for each day in year. 
        Dataframe has following colummns:

        day | margin_top | margin_right | margint_bottom | margin_left
    """
    time = []
    earthsun = []

    imgs_paths = glob.glob(path)

    #length = len(path) - 29

    for img in imgs_paths:
        date = get_date(img)
        time.append(date)

    time.sort(key=lambda date: datetime.strptime(date, "%Y/%m/%d"))

    sun = ephem.Sun()

    # find sun to earth distance
    for thetime in time:
        sun.compute(ephem.Date(thetime))
        earthsun.append(sun.earth_distance)

    time_distance = list(zip(time, earthsun))
    df_distances = pd.DataFrame(time_distance, columns=['time', 'distance'])

    m_top = interp1d([0.9831, 1.0168], [base_margin_top, base_margin_top + 14])
    m_right = interp1d([0.9831, 1.0168], [base_margin_right, base_margin_right + 14])
    m_bottom = interp1d([0.9831, 1.0168], [base_margin_bottom, base_margin_bottom + 14])
    m_left = interp1d([0.9831, 1.0168], [base_margin_left, base_margin_left + 14])

    namapovane_top_arr = []
    namapovane_right_arr = []
    namapovane_bottom_arr = []
    namapovane_left_arr = []


    for item in df_distances.distance:
        namapovane_top = float(m_top(item))
        namapovane_right = float(m_right(item))
        namapovane_bottom = float(m_bottom(item))
        namapovane_left = float(m_left(item))

        namapovane_top_arr.append(round(namapovane_top, 4))
        namapovane_right_arr.append(round(namapovane_right, 4))
        namapovane_bottom_arr.append(round(namapovane_bottom, 4))
        namapovane_left_arr.append(round(namapovane_left, 4))

    df_distances["mask_top"] = namapovane_top_arr
    df_distances["mask_right"] = namapovane_right_arr
    df_distances["mask_bottom"] = namapovane_bottom_arr
    df_distances["mask_left"] = namapovane_left_arr

    return df_distances


# save path musi koncit na /.
def crop_sun(image_name, save_path, mask_top, mask_right, mask_bottom, mask_left):
    """creates white background on image. This preprocesing is needed for 195A images to semgent coronal holes.
    White "background" is created by putting white color in front sun's disk image and draw transparent pieslice on it with suns size.

    Args:
        image_name (string): path to image

        save_path (string): path where to save image

        mask_top (int): margin of pieslice from top

        mask_right (int): margin of pieslice from right

        mask_bottom (int): margin of pieslice from bottom

        mask_left (int): margin of pieslice from left
    """
    img = Image.open(image_name)
    h, w = img.size

    # creating mask
    mask = Image.new('L', [h, w], 0)
    draw = ImageDraw.Draw(mask)
    draw.pieslice([mask_left, mask_top, w - mask_right, w - mask_bottom], 0, 360, fill=255)

    # white foreground
    white_bg = Image.new("RGBA", [h, w], "WHITE")

    image_name_clean = image_name[-29:]
    download_path = save_path + image_name_clean[:len(image_name_clean) - 3] + "png"

    # glue original image, white_background, and transparent mask togheter
    Image.composite(img, white_bg, mask).save(download_path)


def get_date(path):
    """get date from filename path of image. 

    Args:
        path (string): path to file 

    Returns:
        string: string in format %Y/%m/%d
    """
    date_str = path.split("/")[-1].split("_")[0]
    dt = datetime.datetime.strptime(date_str, "%Y%m%d")
    return dt.strftime("%Y/%m/%d")
