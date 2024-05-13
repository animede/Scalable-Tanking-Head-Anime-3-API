import numpy as np
import cv2
from PIL import Image
import argparse
from time import sleep
import time
from poser_client_tkhmp_upmp_v1_3_class import TalkingHeadAnimefaceInterface
from poser_generater_v1_3 import TalkingHeadAnimefaceGenerater
import threading
import pickle

#import uvicorn
#from fastapi import FastAPI, Request, Form, UploadFile, HTTPException,File
#from fastapi.responses import HTMLResponse, StreamingResponse,FileResponse,JSONResponse
from flask import Flask,request,jsonify,send_file,render_template,Response

from io import BytesIO

app = Flask(__name__)
#app = FastAPI()


df_url='http://192.168.5.71:8000/generate/'
tkh_url='http://0.0.0.0:8001'
upscr_url='http://0.0.0.0:8008/resr_upscal/'
    
user_id=0 #便宜上設定している。0~20の範囲。必ず設定すること

#Thiの初期化
#Thi=TalkingHeadAnimefaceInterface(tkh_url)  # tkhのホスト　、アップスケールのURLはプロセス開始で指定
#pose_dic_orgの設定。サーバからもらう
#pose_dic_org = Thi.get_init_dic()
#アップスケールとtkhプロセスの開始
#create_upscale = Thi.create_mp_upscale(upscr_url)
#create_tkh = Thi.create_mp_tkh()

"""
def main():

    input_image  = Image.open("kitchen_anime.png")    
    exec_pose(input_image)
"""
#モーション選択
#@app.post("/api/motion_sel/")
#def move_sel(image:UploadFile = File(...),  user_id:int =Form(...),  move_type:str = Form(...), filename:str=Form(...), user_name:str = Form(...)):

global chr_image
global input_image

#For streeming test@app.route("/api/upload_pose_file_exe", methods=["POST"])
@app.route('/api/live',methods=["GET"])
def live_video():

    global chr_image

    image=chr_image
  
    rame_data=pickle.dumps(image,5) #tx_dataはpklデータ
    #response = make_response(img) # レスポンスに画像を設定
    #response.headers.set('Content-Type', request.content_type) # ヘッダ設定
    
    return Response(response=rame_data)


@app.route("/api/motion_sel/", methods=["POST"])
def move_sel():
    global chr_image
    global input_image
    image=request.files["image"]
    user_id=request.form["user_id"]
    move_type=request.form["move_type"]
    filename=request.form["filename"]
    user_name=request.form["user_name"]

    input_image = image.read()
    # バイトデータをPILイメージオブジェクトに変換
    image_stream = BytesIO(input_image)
    input_image  = Image.open(image_stream)

    chr_image=input_image
    
    #サンプル 1　inference_dic() 　poseはDICT形式で直接サーバを呼ぶ　イメージは事前ロード　  DICT形式で必要な部分のみ選んで連続変化させる
    if move_type=="test1":

        gen_move_th= threading.Thread(target=gen_move,daemon=True)
        gen_move_th.start()

    return jsonify({'message':"ok"})
        
def gen_move():
        global input_image
        df_url='http://192.168.5.71:8000/generate/'
        tkh_url='http://0.0.0.0:8001'
        upscr_url='http://0.0.0.0:8008/resr_upscal/'
        Thi=TalkingHeadAnimefaceInterface(tkh_url)  # tkhのホスト　、アップスケールのURLはプロセス開始で指定
        Thi.create_mp_upscale(upscr_url)
        Thi.create_mp_tkh()
        pose_dic_org = Thi.get_init_dic()

        
        fps=20
        #mode="breastup" #  "breastup" , "waistup" , upperbody" , "full"
        mode="waistup" #  "breastup" , "waistup" , upperbody" , "full"
        #mode=[55,155,200,202] #[top,left,hight,whith]
        scale=2 # 2/4/8
        div_count=300
        #print("*****file_path=",file_path)
        #input_image = Image.open(file_path)#<<<<<<<<<<<<<<<<<<<<<<
        #input_image.show()
        input_image=Thi.image_2_form(input_image, "pil")
        #input_image.show()
        imge = np.array(input_image)
        imge = cv2.cvtColor(imge, cv2.COLOR_RGBA2BGRA)
        result_out_image = imge
        #cv2.imshow("image",imge)
        #cv2.waitKey() #ここで一旦止まり、キー入力で再開する
        img_number=Thi.load_img(input_image,user_id) # 画像のアップロード
        pose_dic_org={"eyebrow":{"menue":"happy","left":0.0,"right":0.0},
                  "eye":{"menue":"wink","left":0.0,"right":0.0},
                  "iris_small":{"left":0.0,"right":0.0},
                  "iris_rotation":{"x":0.0,"y":0.0},
                  "mouth":{"menue":"aaa","val":0.0},
                  "head":{"x":0.0,"y":0.0},
                  "neck":0.0,
                  "body":{"y":0.0,"z":0.0},
                  "breathing":0.0,
                  }

        Tkg=TalkingHeadAnimefaceGenerater(Thi,img_number,user_id,mode,scale,fps)
        #ポーズデータ生成プロセスのスタート
        Tkg.start_mp_generater_process()
        
        #pose_dic=pose_dic_org #Pose 初期値
        current_pose_list=[]
        move_time=div_count/2*(1/fps)
        current_pose_dic=pose_dic_org #Pose 初期値
        
        #Head pose　動作開始
        Tkg.pose_head(0.0, 3.0, 3.0, move_time, current_pose_dic)#head_x,head_y,neck,time,current_pose_dic
        #Head body　動作開始
        Tkg.pose_body(3.0, 3.0, 3.0, move_time, current_pose_dic)#body_y, body_z, breathing,time,current_pose_dic

        #auto_eye_blink_start = Tkg.mp_auto_eye_blink_start(1,2)
        
        mouth_list=["aaa","iii","uuu","eee","ooo","aaa"]
        mouth_pointer=0
        for i in range(int(div_count/2)):
            start_time=time.time()
            # mouthe pose
            mouthe_menue = mouth_list[mouth_pointer]
            if (i==50 or i==60 or i==70 or i==80 or i==100):
                mouth_menue = mouth_list[mouth_pointer]
                Tkg.pose_mouth(mouth_menue, 1.0, 0.1, current_pose_dic)
                mouth_pointer +=1
            if (i==130):
                Tkg.pose_mouth("aaa", 0.0, 0.1, current_pose_dic)
            # mabataki pose
            if (i==20 or i==50):
                Tkg.pose_wink("b", 0.15,current_pose_dic)#l_r,time
            # wink pose
            if (i==10 or i==30):
                Tkg.pose_wink("l", 0.2,current_pose_dic)#l_r,time
            if (i==65):
                Tkg.pose_wink("r", 0.2,current_pose_dic)#l_r,time
            # iris pose
            if (i==5 or i==75):
                Tkg.pose_iris(1.0, 0.0, 0.1,current_pose_dic)#small,rotation,time
            if (i==25 or i==85):
                Tkg.pose_iris(0.0, 0.0, 0.15,current_pose_dic)#small,rotation,time
            if (i==140):
                Tkg.pose_face("happy", 0.0, "happy_wink", 0.0, 0.5,current_pose_dic)#happy :eyebrow_menue, eyebrow, eye_menue, eye, time,current_pose_dic
            #画像の取得
            result_out_image, current_pose_dic = Tkg.get_image()
            chr_image=result_out_image
            
            #cv2.imshow("Loaded image",result_out_image)
            #cv2.waitKey(1)

            print("frame time =",'{:.2f}'.format((1/fps - (time.time()-start_time))*1000),"mS")
            if (1/fps - (time.time()-start_time))>0:
                sleep(1/fps - (time.time()-start_time))
            else:
                print("Remain time is minus")
            print("Genaration time=",(time.time()-start_time)*1000,"mS")
          
        """
        #Head pose　動作開始
        Tkg.pose_head(0.0, -3.0, 0.0, move_time, current_pose_dic)#head_x,head_y,neck,time,current_pose_dic
        # body pose 動作開始
        Tkg.pose_body(-6.0, -3.0, -3.0, move_time, current_pose_dic)#body_y, body_z, breathing,time,current_pose_dic
        for i in range(int(div_count/2)):
            start_time=time.time()
            #Emotion 指定　→ "happy" #喜 "angry" #怒 "sorrow" #哀 "relaxed" #楽 "smile" #微笑む "laugh" #笑う "surprised" #驚く
            if i==20:
                Tkg.pose_emotion("happy",0.5, current_pose_dic)
            if i==60:
                Tkg.pose_emotion("angry", 0.5, current_pose_dic)
            if i==100:
                Tkg.pose_emotion("sorrow", 0.5, current_pose_dic)
            if i==140:
                Tkg.pose_emotion("relaxed", 0.5, current_pose_dic)
            #画像の取得
            result_out_image, current_pose_dic = Tkg.get_image()
            
            cv2.imshow("Loaded image",result_out_image)
            cv2.waitKey(1)
            #cv2.imwrite("image1/image2"+str(i+2000)+".jpg",result_out_image)
            print("1/fps - (time.time()-start_time)=",1/fps - (time.time()-start_time))
            if (1/fps - (time.time()-start_time))>0:
                sleep(1/fps - (time.time()-start_time))
            else:
                print("Remain time is minus")
            print("Genaration time=",(time.time()-start_time)*1000,"mS")

        #Head pose　動作開始
        Tkg.pose_head(0.0, 0.0, 0.0, move_time, current_pose_dic)#head_x,head_y,neck,time,current_pose_dic
        # body pose 動作開始
        Tkg.pose_body(0.0, 0.0, 0.0, move_time, current_pose_dic)#body_y, body_z, breathing,time,current_pose_dic

        #Tkg.mp_auto_eye_blink_start(1,3)
        for i in range(int(div_count/2)):
            start_time=time.time()
            #Emotion 指定　→ "happy" #喜 "angry" #怒 "sorrow" #哀 "relaxed" #楽 "smile" #微笑む "laugh" #笑う "surprised" #驚く
            if i==20:
                Tkg.pose_emotion("laugh", 0.5, current_pose_dic)
            if i==60:
                Tkg.pose_emotion("surprised", 0.2, current_pose_dic)   
            if i==800:
                Tkg.pose_emotion("smile", 0.5, current_pose_dic)
            if i==100:
                Tkg.pose_face("happy", 0.0, "happy_wink", 0.0, 0.5,current_pose_dic)#happy :eyebrow_menue, eyebrow, eye_menue, eye, time,current_pose_dic
                Tkg.pose_mouth("aaa", 0.0, 0.5, current_pose_dic)
            #画像の取得
            result_out_image, current_pose_dic = Tkg.get_image()

            chr_image=result_out_image
            
            #cv2.imshow("Loaded image",result_out_image)
            #cv2.waitKey(1)
            #cv2.imwrite("image1/image3"+str(i+3000)+".jpg",result_out_image)
            print("1/fps - (time.time()-start_time)=",1/fps - (time.time()-start_time))
            if (1/fps - (time.time()-start_time))>0:
                sleep(1/fps - (time.time()-start_time))
            else:
                print("Remain time is minus")
            print("Genaration time=",(time.time()-start_time)*1000,"mS")
        """
        #サブプロセスの終了
        Thi.up_scale_proc_terminate()
        Thi.tkh_proc_terminate()
        Tkg.mp_generater_process_terminate()

        print("end of move")

        #return jsonify({'message':"ok"})

if __name__ == "__main__":
    app.run(debug=True,port=8011)

