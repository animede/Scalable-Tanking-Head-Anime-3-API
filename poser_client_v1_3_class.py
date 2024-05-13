import numpy as np
import cv2
from PIL import Image
import time
import requests
import pickle

#generation Classのバリエーション
#
# inference(self,input_img,current_pose):                    #pose=リポジトリの形式、イメージは毎回ロード
# inference_img(self,current_pose,img_number,user_id):       # pose=リポジトリの形式  イメージは事前ロード,複数画像対応
# inference_pos(self,packed_current_pose,img_number,user_id):# pose=パック形式　イメージは事前ロード,複数画像対応
# inference_dic(self,current_dic,img_number,user_id):        # pose=Dict形式 イメージは事前ロード,複数画像対応

# ユーティリティClass
# get_pose(self,pose_pack):        #パック形式 =>リポジトリの形式変換
# get_init_dic(self):              #Dict形式の初期値を得る
# get_pose_dic(self,dic):          #Dict形式 => リポジトリの形式変換
# load_img(self,input_img,user_id):# 画像をVRAMへ登録

class TalkingHeadAnimefaceInterface():
    def __init__(self,host):
        userid=0
        self.url=host

    def get_init_dic(self):
        response = requests.post(self.url+"/get_init_dic/") #リクエスト
        if response.status_code == 200:
            pose_data = response.content
            org_dic =(pickle.loads(pose_data))#元の形式にpickle.loadsで復元
        return org_dic  
      
    def get_pose(self,pose_pack):
        #-----パック形式
        #0  eyebrow_dropdown: str :            "troubled", "angry", "lowered", "raised", "happy", "serious"
        #1  eyebrow_leftt, eyebrow_right:      float:[0.0,0.0]
        #2  eye_dropdown: str:                 "wink", "happy_wink", "surprised", "relaxed", "unimpressed", "raised_lower_eyelid"
        #3  eye_left, eye_right :              float:[0.0,0.0]
        #4  iris_small_left, iris_small_right: float:[0.0,0.0]
        #5 iris_rotation_x, iris_rotation_y : float:[0.0,0.0]
        #6  mouth_dropdown: str:               "aaa", "iii", "uuu", "eee", "ooo", "delta", "lowered_corner", "raised_corner", "smirk"
        #7  mouth_left, mouth_right :          float:[0.0,0.0]
        #8  head_x, head_y :                   float:[0.0,0.0]
        #9  neck_z,                            float
        #10 body_y,                            float
        #11 body_z:                            float
        #12 breathing:                         float
        #
        # Poseの例
        # pose=["happy",[0.5,0.0],"wink", [i/50,0.0], [0.0,0.0], [0.0,0.0],"ooo", [0.0,0.0], [0.0,i*3/50],i*3/50, 0.0, 0.0, 0.0]
        
        pose_pack_pkl = pickle.dumps(pose_pack, 5)
        files = {"pose":("pos.dat",pose_pack_pkl, "application/octet-stream")}#listで渡すとエラーになる
        response = requests.post(self.url+"/get_pose/", files=files) #リクエスト
        if response.status_code == 200:
            pose_data = response.content
            pose =(pickle.loads(pose_data))#元の形式にpickle.loadsで復元
            result = response.status_code
        return pose  
  
    def get_pose_dic(self,dic):
            #サンプル Dict形式
            #"mouth"には2種類の記述方法がある"lowered_corner"と”raised_corner”は左右がある
            #  "mouth":{"menue":"aaa","val":0.0},
            #  "mouth":{"menue":"lowered_corner","left":0.5,"right":0.0},　これはほとんど効果がない
            #
            #pose_dic={"eyebrow":{"menue":"happy","left":0.5,"right":0.0},
            #        "eye":{"menue":"wink","left":0.5,"right":0.0},
            #        "iris_small":{"left":0.0,"right":0.0},
            #        "iris_rotation":{"x":0.0,"y":0.0},
            #        "mouth":{"menue":"aaa","val":0.7},
            #        "head":{"x":0.0,"y":0.0},
            #        "neck":0.0,
            #        "body":{"y":0.0,"z":0.0},
            #        "breathing":0.0
            #        }
            
        current_dic = pickle.dumps(dic, 5)
        files = {"pose":("pos.dat",current_dic, "application/octet-stream")}#listで渡すとエラーになる
        response = requests.post(self.url+"/get_pose_dic/", files=files) #リクエスト
        if response.status_code == 200:
            pose_data = response.content
            pose =(pickle.loads(pose_data))#元の形式にpickle.loadsで復元
            result = response.status_code
        return pose   

    def load_img(self,input_img,user_id):
        print("load_img")
        images_data = pickle.dumps(input_img, 5) 
        files = {"image": ("img.dat",  images_data, "application/octet-stream")}
        data = {"user_id": user_id}
        response = requests.post(self.url+"/load_img/", files=files, data=data) #リクエスト送信
        if response.status_code == 200:
            response_data = response.json()
            print("response_data =",response_data)
            img_number=response_data["img_number"]
        else:
            img_number=-1
        return img_number

    def inference(self,input_img,current_pose,out="pil"):#基本イメージ生成、イメージは毎回ロード
        start_time=time.time()
        images_data = pickle.dumps(input_img, 5)
        current_pose2 = pickle.dumps(current_pose, 5)
        files = {"image": ("img.dat",images_data, "application/octet-stream"),
                 "pose":("pos.dat",current_pose2, "application/octet-stream"),
                 "out":("out.dat", out, "application/octet-stream")}#listで渡すとエラーになる
        response = requests.post(self.url+"/inference_org/", files=files) #リクエスト
        if response.status_code == 200:
            image_data = response.content
            image =(pickle.loads(image_data))#元の形式にpickle.loadsで復元
            result = response.status_code
        return result, image
    
    def inference_pos(self,packed_pose,img_number,user_id,out="pil"):#イメージは事前ロード
        packed_pose = pickle.dumps(packed_pose, 5)
        files={"pose":("pos.dat",packed_pose, "application/octet-stream"),}
              # "img_number":img_number,
              # "user_id": user_id,}#listで渡すとエラーになる
        data = {"user_id": user_id,"img_number":img_number,"out":out}
        response = requests.post(self.url+"/inference_pos/", files=files, data=data) #リクエスト送信
        if response.status_code == 200:
            image_data = response.content
            image =(pickle.loads(image_data))#元の形式にpickle.loadsで復元
            result = response.status_code
        return result, image

    def inference_dic(self,current_dic,img_number,user_id,out="pil"):#イメージは事前ロード
        data = {"img_number":img_number,"user_id": user_id,"out":out}
        current_dic2 = pickle.dumps(current_dic, 5)
        files={"pose":("pos.dat",current_dic2, "application/octet-stream")}#listで渡すとエラーになる
        response = requests.post(self.url+"/inference_dic/", data=data,files=files) #リクエスト送信
        if response.status_code == 200:
            image_data = response.content
            image =(pickle.loads(image_data))#元の形式にpickle.loadsで復元
            result = response.status_code
        return result, image
        
    def inference_img(self,current_pose,img_number,user_id,out="pil"):#イメージ事前ロード用生成 イメージは事前ロード
        data = {"current_pose":current_pose,"img_number":img_number,"user_id": user_id,"out":out}
        response = requests.post(self.url+"/inference_img/", data=data) #リクエスト送信
        if response.status_code == 200:
            image_data = response.content
            image =(pickle.loads(image_data))#元の形式にpickle.loadsで復元
            result = response.status_code
        return result, image


