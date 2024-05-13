import numpy as np
import pickle
import cv2
from PIL import Image
import argparse
from time import sleep
import time
from poser_client_mp_v1_3_class import TalkingHeadAnimefaceInterface

import subprocess
from subprocess import Popen
from multiprocessing import Process
import multiprocessing
import threading

from tkh_up_scale import upscale


#PIL形式の画像を動画として表示
def image_show(imge):
    imge = np.array(imge)
    imge = cv2.cvtColor(imge, cv2.COLOR_RGBA2BGRA)
    cv2.imshow("Loaded image",imge)
    cv2.waitKey(1)

def main():
    parser = argparse.ArgumentParser(description='Talking Head')
    parser.add_argument('--filename','-i', default='000002.png', type=str)
    parser.add_argument('--test', default=0, type=str)
    parser.add_argument('--host', default='http://0.0.0.0:8001', type=str)
    args = parser.parse_args()

    esr_host="0.0.0.0"    # サーバーIPアドレス定義
    esr_port=8008          # サーバー待ち受けポート番号定義
    url="http://" + esr_host + ":" + str(esr_port) + "/resr_upscal/"
    
    test =args.test
    print("TEST=",test)
    filename =args.filename
    global_out_image = Image.open(filename)

    Thi=TalkingHeadAnimefaceInterface(args.host)

    user_id=0

    #pose_dic_orgの設定。サーバからもらう
    pose_dic_org = Thi.get_init_dic()

    #アップスケールプロセスの開始
    Thi.create_mp_upscale(url)
            
    #サンプル 7　inference_img() poseはパック形式形式をリポジトリの形式に変換 イメージは事前ロード,パック形式で連続変化させる  
    if test=="7":
        fps=50
        #mode="breastup"
        mode="waistup"
        mode=[55,155,200,202]
        scale=4
        input_image = Image.open(filename)
        imge = np.array(input_image)
        imge = cv2.cvtColor(imge, cv2.COLOR_RGBA2BGRA)
        result_out_image = imge
        cv2.imshow("image",imge)
        cv2.waitKey()
        img_number=Thi.load_img(input_image,user_id)
        print("img_number=",img_number)
        for i in range(50):
            start_time=time.time()
            packed_current_pose=[
                "happy",[0.5,0.0],"wink", [i/50,0.0], [0.0,0.0], [0.0,0.0],"ooo", [0.0,0.0], [0.0,i*3/50],i*3/50, 0.0, 0.0, 0.0]
            #current_pose=Thi.get_pose(packed_current_pose) #packed_pose=>current_pose
            result_out_image = Thi.mp_pack2image_frame(result_out_image,packed_current_pose,img_number,user_id,mode,scale,fps)

            print("Genaration time=",(time.time()-start_time)*1000,"mS")
            cv2.imshow("Loaded image",result_out_image)
            cv2.waitKey(1)

        for i in range(100):
            start_time=time.time()
            packed_current_pose=[
                "happy", [0.5,0.0], "wink",[1-i/50,0.0], [0.0,0.0], [0.0,0.0], "ooo", [0.0,0.0], [0.0,3-i*3/50], 3-i*3/50, 0.0, 0.0, 0.0,]
            #current_pose=Thi.get_pose(packed_current_pose)#packed_current_pose==>current_pose
            result_out_image = Thi.mp_pack2image_frame(result_out_image,packed_current_pose,img_number,user_id,mode,scale,fps)
            
            print("Genaration time=",(time.time()-start_time)*1000,"mS")
            cv2.imshow("Loaded image",result_out_image)
            cv2.waitKey(1)
            
        for i in range(100):
            start_time=time.time()
            packed_current_pose=[
                "happy", [0.5,0.0], "wink", [i/100,i/100], [0.0,0.0], [0.0,0.0], "ooo", [0.0,0.0], [0.0,-3+i*3/50], -3+i*3/50,0.0, 0.0,0.0,]
            current_pose=Thi.get_pose(packed_current_pose)#packed_current_pose==>current_pose
            result_out_image = Thi.mp_pose2image_frame(result_out_image,current_pose,img_number,user_id,mode,scale,fps)
            
            print("Genaration time=",(time.time()-start_time)*1000,"mS")
            cv2.imshow("Loaded image",result_out_image)
            cv2.waitKey(1)
            
        for i in range(100):
            start_time=time.time()
            packed_current_pose=[
                "happy", [0.5,0.0], "wink", [0.0,0.0], [0.0,0.0], [0.0,0.0], "ooo",  [0.0,0.0], [0.0,3-i*3/100],  3-i*3/100,  0.0, 0.0, 0.0,]
            current_pose=Thi.get_pose(packed_current_pose) #packed_current_pose==>current_pose
            result_out_image = Thi.mp_pose2image_frame(result_out_image,current_pose,img_number,user_id,mode,scale,fps)

            print("Genaration time=",(time.time()-start_time)*1000,"mS")
            cv2.imshow("Loaded image",result_out_image)
            cv2.waitKey(1)
        cv2.imshow("Loaded image",result_out_image)
        cv2.waitKey(5000)

    #サンプル 8　inference_dic() 　poseはDICT形式で直接サーバを呼ぶ　イメージは事前ロード　  DICT形式で必要な部分のみ選んで連続変化させる
    if test=="8":
        fps=35
        #mode="breastup"
        mode="waistup"
        scale=8
        div_count=30
        input_image = Image.open(filename)
        imge = np.array(input_image)
        imge = cv2.cvtColor(imge, cv2.COLOR_RGBA2BGRA)
        result_out_image = imge
        cv2.imshow("image",imge)
        cv2.waitKey()
        img_number=Thi.load_img(input_image,user_id)
        pose_dic=pose_dic_org #Pose 初期値
        current_pose_list=[]
        for i in range(int(div_count/2)):
            start_time=time.time()
            current_pose_dic=pose_dic
            current_pose_dic["eye"]["menue"]="wink"
            current_pose_dic["eye"]["left"]=i/(div_count/2)
            current_pose_dic["head"]["y"]=i*3/(div_count/2)
            current_pose_dic["neck"]=i*3/(div_count/2)
            current_pose_dic["body"]["y"]=i*5/(div_count/2)

            result_out_image = Thi.mp_dic2image_frame(result_out_image,current_pose_dic,img_number,user_id,mode,scale,fps)
            
            print("Genaration time=",(time.time()-start_time)*1000,"mS")
            cv2.imshow("Loaded image",result_out_image)
            cv2.waitKey(1)
        for i in range(div_count):
            start_time=time.time()
            current_pose_dic["eye"]["left"]=1-i/(div_count/2)
            current_pose_dic["head"]["y"]=3-i*3/(div_count/2)
            current_pose_dic["neck"]=3-i*3/(div_count/2)
            current_pose_dic["body"]["y"]=5-i*5/(div_count/2)

            result_out_image = Thi.mp_dic2image_frame(result_out_image,current_pose_dic,img_number,user_id,mode,scale,fps)
            
            print("Genaration time=",(time.time()-start_time)*1000,"mS")
            cv2.imshow("Loaded image",result_out_image)
            cv2.waitKey(1)
        for i in range(div_count):
            start_time=time.time()
            current_pose_dic["eye"]["left"]=i/div_count
            current_pose_dic["eye"]["right"]=i/div_count
            current_pose_dic["head"]["y"]=-3+i*3/(div_count/2)
            current_pose_dic["neck"]=-3+i*3/(div_count/2)
            current_pose_dic["body"]["z"]=i*3/div_count
            current_pose_dic["body"]["y"]=-5+i*5/(div_count/2)

            result_out_image = Thi.mp_dic2image_frame(result_out_image,current_pose_dic,img_number,user_id,mode,scale,fps)
            
            print("Genaration time=",(time.time()-start_time)*1000,"mS")
            cv2.imshow("Loaded image",result_out_image)
            cv2.waitKey(1)
        for i in range(div_count):
            start_time=time.time()
            current_pose_dic["eye"]["left"]=0.0
            current_pose_dic["eye"]["right"]=0.0
            current_pose_dic["head"]["y"]=3-i*3/(div_count/2)
            current_pose_dic["neck"]=3-i*3/(div_count/2)
            current_pose_dic["body"]["z"]=3-i*3/div_count
            current_pose_dic["body"]["y"]=5-i*5/(div_count/2)

            result_out_image = Thi.mp_dic2image_frame(result_out_image,current_pose_dic,img_number,user_id,mode,scale,fps)
            
            print("Genaration time=",(time.time()-start_time)*1000,"mS")
            cv2.imshow("Loaded image",result_out_image)
            cv2.waitKey(1)
        cv2.imshow("Loaded image",result_out_image)
        cv2.waitKey(5000)

    #サブプロセスの終了
    Thi.proc_terminate()
    print("end of test")
            
if __name__ == "__main__":
    main()
