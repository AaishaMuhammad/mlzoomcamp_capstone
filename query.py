import requests
import urllib.request
from os import remove
import argparse

# url = "http://localhost:8080/classify" # <----- Uncomment this for testing Local Deploy, comment the below line
url = "https://kitchenware-model-pwwdruaj5q-uc.a.run.app/classify" # <----- Uncomment this to test Cloud Deploy, comment the above line

parser = argparse.ArgumentParser(description="Pass a URL to a .jpg image.")
parser.add_argument('image_url', type=str, help="input a url to a jpg image")
args = parser.parse_args()

img_path = args.image_url

urllib.request.urlretrieve(img_path, 'image.jpg')

with open('image.jpg', 'rb') as img_file:
    payload = {"img": img_file}
    print(requests.post(url, files=payload).json()) 

remove('./image.jpg')

