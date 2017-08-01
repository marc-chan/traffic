from io import BytesIO
import requests
import numpy as np
import datetime
from PIL import Image
from config import config

##Apply for your own API key @ https://data.gov.sg
API_KEY = config["C_KEY"]
URL = "https://api.data.gov.sg/v1/transport/traffic-images"

def convert_timestring(dt):
    #Format required for api query: YYYY-MM-DD[T]HH:MM:SS+08:00'
    return dt.strftime("%Y-%m-%dT%X+08:00")

def get_one_traffic_data(date_time):
    r = requests.get(URL, params={"date_time": date_time}, headers={"api-key": API_KEY})
    data = r.json()
    return data

def find_image_url(camera_id, data):
    camera_id = str(camera_id)
    c_arr = data["items"][0]["cameras"]
    for c in c_arr:
        if c["camera_id"] == camera_id:
            return c["image"]
    print("camera image not found")

def pad_image(img):
    longer_side = max(img.size)
    horizontal_padding = (longer_side - img.size[0]) / 2
    vertical_padding = (longer_side - img.size[1]) / 2
    pad_img = img.crop(
        (
            -horizontal_padding,
            -vertical_padding,
            img.size[0] + horizontal_padding,
            img.size[1] + vertical_padding
        )
    )
    return pad_img

def load_one_image(camera_id,date_time=datetime.datetime.now(),offset=0): #input a python datetime object
    ##Queries the API, download image, return np array
    date_string = convert_timestring(date_time-datetime.timedelta(minutes=offset))
    url = find_image_url(camera_id,get_one_traffic_data(date_string))
    response = requests.get(url, stream=True)
    img = Image.open(BytesIO(response.content))
    imgarr = np.array(img)
    return imgarr
