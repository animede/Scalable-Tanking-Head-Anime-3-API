import numpy as np
import cv2
from PIL import Image
import argparse
from time import sleep
import time
from io import BytesIO

import requests

filename="kitchen_anime.png"
input_image = Image.open(filename)

# PILイメージをバイト列に変換
buffer = BytesIO()
input_image.save(buffer, format='PNG')  # 保存フォーマットは必要に応じて変更
image_bytes = buffer.getvalue()

data= {"user_id":0,
       "move_type":"test1",
       "filename":filename,
       "user_name":"test"}
files={"image": ("image.png", image_bytes, 'image/png')}
response = requests.post( "http://0.0.0.0:8011/api/motion_sel/", files=files, data=data) #リクエスト
if response.status_code == 200:
            response_data = response.json()
            print("response_data =",response_data)
            #img_number=response_data["img_number"]
else:
            img_number=-1


