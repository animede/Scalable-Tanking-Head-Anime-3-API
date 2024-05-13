import numpy as np
import cv2
from PIL import Image
import argparse
from time import sleep
import time
from poser_client_tkhmp_upmp_v1_3_class import TalkingHeadAnimefaceInterface
from poser_generater_v1_3 import TalkingHeadAnimefaceGenerater
import os
import signal
import pickle

import uvicorn
from fastapi import FastAPI, Request, Form, UploadFile, HTTPException,File
from fastapi.responses import HTMLResponse, StreamingResponse,FileResponse,JSONResponse
from starlette.responses import Response

from io import BytesIO

app = FastAPI()

global stream_image
global input_image

df_url='http://192.168.5.71:8000/generate/'
tkh_url='http://0.0.0.0:8001'
upscr_url='http://0.0.0.0:8008/resr_upscal/'
    
user_id=0 #便宜上設定している。0~20の範囲。必ず設定すること

#Thiの初期化
Thi=TalkingHeadAnimefaceInterface(tkh_url)  # tkhのホスト　、アップスケールのURLはプロセス開始で指定
#pose_dic_orgの設定。サーバからもらう
#pose_dic_org = Thi.get_init_dic()

#アップスケールとtkhプロセスの開始
#create_upscale = Thi.create_mp_upscale(upscr_url)
create_tkh = Thi.create_mp_tkh()

"""
def main():

    input_image  = Image.open("kitchen_anime.png")    
    exec_pose(input_image)
"""
@app.get("/api/live")
def live_video():
    global stream_image
    #try:
    #    image = stream_image
    #except:
    #    image = Image.open("white.png")
        
    image = stream_image
    frame_data = pickle.dumps(image, 4)  # tx_dataはpklデータ
    return Response(content=frame_data, media_type="application/octet-stream")

#イメージのアップロード
@app.post("/api/upload_img/")
async def motion_sel(image:UploadFile = File(...),  user_id:int =Form(...),  filename:str=Form(...), user_name:str = Form(...)):
    global input_image
    input_image = await image.read()
    # バイトデータをPILイメージオブジェクトに変換
    image_stream = BytesIO(input_image )
    input_image  = Image.open(image_stream)

#イメージのダウンロード
@app.post("/api/motion_sel/")
def motion_sel(user_id:int =Form(...),  move_type:str = Form(...), filename:str=Form(...), user_name:str = Form(...)):
    global stream_image
    global input_image

    df_url='http://192.168.5.71:8000/generate/'
    tkh_url='http://0.0.0.0:8001'
    upscr_url='http://0.0.0.0:8008/resr_upscal/'
    #アップスケールとtkhプロセスの開始
    #Thi=TalkingHeadAnimefaceInterface(tkh_url)  # tkhのホスト　、アップスケールのURLはプロセス開始で指定
    upscal_pid=Thi.create_mp_upscale(upscr_url)
    #Thi.create_mp_tkh()
    pose_dic_org = Thi.get_init_dic()
    #サンプル 1　inference_dic() 　poseはDICT形式で直接サーバを呼ぶ　イメージは事前ロード　  DICT形式で必要な部分のみ選んで連続変化させる
    if move_type=="test1":

        fps=20
        #mode="breastup" #  "breastup" , "waistup" , upperbody" , "full"
        mode="waistup" #  "breastup" , "waistup" , upperbody" , "full"
        #mode=[55,155,200,202] #[top,left,hight,whith]
        scale=2 # 2/4/8
        div_count=300

        input_image.show()
        input_image=Thi.image_2_form(input_image, "pil")

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
        pid=Tkg.start_mp_generater_process()
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ pid=",pid)
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
            stream_image=result_out_image
            
            cv2.imshow("Loaded image",result_out_image)
            cv2.waitKey(1)

            print("1/fps - (time.time()-start_time)=",1/fps - (time.time()-start_time))
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
            
            cv2.imshow("Loaded image",result_out_image)
            cv2.waitKey(1)
            #cv2.imwrite("image1/image3"+str(i+3000)+".jpg",result_out_image)
            print("1/fps - (time.time()-start_time)=",1/fps - (time.time()-start_time))
            if (1/fps - (time.time()-start_time))>0:
                sleep(1/fps - (time.time()-start_time))
            else:
                print("Remain time is minus")
            print("Genaration time=",(time.time()-start_time)*1000,"mS")


        """
        #cv2.imshow("Loaded image",result_out_image)
        #cv2.waitKey(5)
        #cv2.waitKey(1000)
    cv2.destroyWindow("Loaded image")
    cv2.waitKey(1)
    #サブプロセスの終了
    #Thi.up_scale_proc_terminate()
    #Thi.tkh_proc_terminate()
    #Tkg.mp_generater_process_terminate()
    os.kill(pid, signal.SIGKILL)
    os.kill(upscal_pid,signal.SIGKILL)
    #Tkg.mp_auto_eye_blink_teminate()
    #Tkg.mp_all_proc_terminate()
    sleep(2)
    print("end of process")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8011)
"""
if __name__ == "__main__":
    main()
"""
