import streamlit as st
import datetime
from datetime import timedelta
from PIL import Image
import glob
import scss_model
import os
import settings as Settings


def make_room(n):
    """generates white space on streamlit site. Basically it is an aleternative to <br> from html.

    Args:
        n (int): how many lines it should break
    """
    for i in range(n):
        st.write("")


def find_image(date, event, cropped=False):
    """Finds and returns the path to an image from ../data/___ file. based on the provided date and event, either "CH" to look for
    195A images (best to spot coronal holes) or "AR" to look for 171A images (best to spot active regions).

    Args:
        date (datetime): date for which function will find image. example: 2002/01/31

        event (string): "CH" or "AR"

        cropped (bool, optional): to find image with cropped background of suns disk, only available for "CH" event. Defaults to False.

    Returns:
        string: path to image
    """

    date = date.strftime('%Y%m%d')

    if event == "CH":
        if cropped:
            # Look for cropped 195A images
            imgs = glob.glob(f"{Settings.IMAGES_195_CROPPED}*.png")

        else:
            # Look for standard 195A images
            imgs = glob.glob(f"{Settings.IMAGES_195}*.jpg")
    else:
        # Look for 171A images for active regions
        imgs = glob.glob(f"{Settings.IMAGES_171}*.png")


    path_to_img = ""

    for path in imgs:
        if date in path:
            path_to_img = path

    if not path_to_img:
        path_to_img = {Settings.MISSING_IMAGE}

    return path_to_img


class passDate:
    """class to pass date from widget input to buttons
    """
    def __init__(self):
        self.date = None

    def setDate(self, date):
        self.date = date

    def getDate(self):
        return self.date


class passDateBack:
    """class to pass date back from buttons to widget
    """
    def __init__(self):
        self.date = None
        self.change = False

    def setDate(self, date):
        self.date = date

    def getDate(self):
        return self.date

    def setSwitch(self, change):
        self.change = change

    def getSwitch(self):
        return self.change

# @st.cache(allow_output_mutation=True)
# @st.cache_resource
@st.cache(allow_output_mutation=True)
def pass_date_init():
    """function to run only once after streamlit webapp has started. Returns initialized classes for passing dates

    Returns:
        passDate(): class to pass  date rom calendar widget input to buttons

        passDateBack(): class to pass date from buttons to widget intput
    """
    return passDate(), passDateBack()

def increment_counter(increment_value=0):
    """this function provides counter for how many times previous or next day button was clicked. 
    This is needed to determine which image to show after user clicked on buttons

    Args:
        increment_value (int, optional): increment session state by 1. Defaults to 0.
    """
    st.session_state.count += 1


def decrement_counter(decrement_value=0):
    """this function provides counter for how many times previous or next day button was clicked. 
    This is needed to determine which image to show after user clicked on buttons

    Args:
        decrement_value (int, optional): decrement session state by 1. Defaults to 0.
    """
    st.session_state.count -= 1


# -----------------------------------------------------
# -------------------------WEBAPP----------------------
# -----------------------------------------------------


st.set_page_config(page_title="DL on SOHO", layout="wide")

st.title("Usage of deep learning for segmentation of selective events in solar corona")

make_room(3)

# page layout to make col1 for text, empty col that makes white space and col2 for image
col1, colEMPTY, col2 = st.columns([0.5, 0.5, 1])

with col1:
    make_room(6)

    # initialize session state
    if 'count' not in st.session_state:
        st.session_state.count = 0


    # initialize classes only once
    pass_date, pass_date_back = pass_date_init()

    option = st.selectbox(
        'choose date selection: ',
        ('date input', 'buttons'))

    if option == "date input":

        if pass_date_back.getSwitch():
            date_start = pass_date_back.getDate()
            st.session_state.count = 0
        else:
            date_start = datetime.date(2011, 1, 1)

        date = st.date_input(
            "Choose a date to display the solar disk",
            date_start,
            min_value=datetime.date(1996, 1, 1),
            max_value=datetime.date(2021, 12, 31),
        )

        pass_date.setDate(date)

    else:
        st.button('Next day ->', on_click=increment_counter, kwargs=dict(increment_value=1))

        st.button('<- Previous day', on_click=decrement_counter, kwargs=dict(decrement_value=1))

        date_button = pass_date.getDate()
        date = date_button + timedelta(days=st.session_state.count)
        pass_date_back.setDate(date)
        pass_date_back.setSwitch(True)

    make_room(2)

    event_option = st.selectbox(
        'choose an event to segment: ',
        ('coronal holes', 'active regions'))
    if event_option == "coronal holes":
        event = "CH"
    else:
        event = "AR"

    # img path load
    path_to_img = find_image(date, event)

    if st.checkbox(f'segment {event}'):
        path_to_img_checked = find_image(date, event, cropped=True)
        # scss model segemtnation
        if "missing" in path_to_img_checked:
            # if there is no image of selected date, segmentation is not happening
            image = Image.open(path_to_img)
            pass
        else:
            image, area_coverage = scss_model.start_segmentation(path_to_img_checked, event)
            st.markdown(f"{event_option} on this image cover ***{area_coverage}%*** of Sun's disk")

    else:
        image = Image.open(path_to_img)

with colEMPTY:
    pass

with col2:
    st.image(image, caption='soho eit')

