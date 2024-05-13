import numpy as np
import pickle
import cv2
from PIL import Image
import argparse
from time import sleep
import time
from poser_client_v1_3_class import TalkingHeadAnimefaceInterface

from tkh_up_scale import upscale
#PIL形式の画像を動画として表示
def image_show(imge):
    imge = np.array(imge)
    imge = cv2.cvtColor(imge, cv2.COLOR_RGBA2BGRA)
    cv2.imshow("Loaded image",imge)
    cv2.waitKey(1)

def main():

    print("TEST")
    
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

    Thi=TalkingHeadAnimefaceInterface(args.host)

    user_id=0

    #pose_dic_orgの設定。サーバからもらう
    pose_dic_org = Thi.get_init_dic()
    
    #pose_dic_orgの設定。自分で書く
    #pose_dic_org={"eyebrow":{"menue":"happy","left":0.0,"right":0.0},
    #          "eye":{"menue":"wink","left":0.0,"right":0.0},
    #          "iris_small":{"left":0.0,"right":0.0},
    #          "iris_rotation":{"x":0.0,"y":0.0},
    #          "mouth":{"menue":"aaa","val":0.0},
    #          "head":{"x":0.0,"y":0.0},
    #          "neck":0.0,
    #          "body":{"y":0.0,"z":0.0},
    #          "breathing":0.0
    #          }
   
    
    #*************************　便利な独自pose形式　*****************************
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
    #
    #-----Dict形式
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

    #*************************リポジトリの引数形式***********************
    #-----　展開したcurrent_poseの形式
    #current_pose= [
    #0  troubled_eyebrow_left=0.0,
    #1  troubled_eyebrow_right=0.0,
    #2  angry_eyebrow_left= 0.0,
    #3  angry_eyebrow_right 0.0,
    #4  lowered_eyebrow_left= 0.0,
    #5  lowered_eyebrow_right 0.0,
    #6  raised_eyebrow_left= 0.0,
    #7  raised_eyebrow_right 0.0,
    #8  happy_eyebrow_left= 0.0,
    #9  happy_eyebrow_right 0.02,
    #10 serious_eyebrow_left= 0.0,
    #11 serious_eyebrow_right=0.0,
    
    #12 eye_wink_left= 0.0,
    #13 eye_wink_right=0.0,
    #14 eye_happy_wink=0.0,
    #15 eye_happy_wink=0.0,
    #16 eye_suprised_left=0.0,
    #17 eye_suprised_right=0.0,
    #18 eye_relaxed_left=0.0,
    #19 eye_relaxed_right=0.0,
    #20 eye_unimpressed_left=0.0,
    #21 eye_unimpressed_right=0.0,
    #22 eye_raised_lower_eyelid_left=0.0,
    #23 eye_raised_lower_eyelid_right=_0.0,
    
    #24 iris_small_left=0.0,
    #25 iris_small_right0.0,
    
    #26 mouth_dropdown_aaa=0.0,
    #27 mouth_dropdown_iii=0.0,
    #28 mouth_dropdown_uuu=0.0,
    #29 mouth_dropdown_eee=0.0,
    #30 mouth_dropdown_ooo=0.0,
    #31 mouth_dropdown_delta=0.0,
    #32 mouth_dropdown_lowered_corner_left=0.0,
    #33 mouth_dropdown_lowered_corner_right=0.0,
    #34 mouth_dropdown_raised_corner_left=0.0,
    #35 mouth_dropdown_raised_corner_right=0.0,
    #36 mouth_dropdown_smirk=0.0,
    
    #37 iris_rotation_x=0.0,
    #38 iris_rotation_y=0.0,
    #39 head_x=0.0,
    #40 head_y=0.0,
    #41 neck_z=0.0,
    #42 body_y=0.0,
    #43 body_z=0.0,
    #44 breathing= 0.0
    #]

    if test=="0":
        user_id=0
        input_image  = Image.open(filename)
        input_image.show()
        image_number = Thi.load_img(input_image,user_id)
        print("image_number=",image_number)


    #サンプル　1　ベタ書き リポジトリの形式　 inference()を使用 イメージは毎回ロード
    if test=="1":  #inference()のテスト
        input_image = Image.open(filename)
        input_image.show()
        #image_number = Thi.load_img(input_image,user_id)
        current_pose = [0.0, 0.0,
                   0.0, 0.0,
                   0.0, 0.0,
                   0.0, 0.0,
                   0.0, 0.0,
                   0.0, 0.0,
                   
                   0.0, 0.5,
                   0.0, 0.0,
                   0.0, 0.0,
                   0.0, 0.0,
                   0.0, 0.0,
                   0.0, 0.0,

                   0.0, 0.0,

                   0.0,
                   0.0,
                   0.0,
                   0.0,
                   0.0,
                   0.0,
                   0.0, 0.0,
                   0.0, 0.0,
                   0.767,
                   0.566,
                   0.626,

                   0.747,
                   0.485,

                   0.444,
                   0.232,
                   
                   0.646,
                   1.0]
        result, out_image=Thi.inference(input_image,current_pose)
        out_image.show()

    #サンプル ２　inference()を使用　パック形式をリポジトリの形式に変換 　イメージは毎回ロード  #packed_pose=>current_pose2
    if test=="2": 
        input_image = Image.open(filename)
        input_image.show()
        packed_pose=["happy", [0.5,0.0], "wink", [1.0,0.0], [0.0,0.0], [0.0,0.0], "ooo", [0.0,0.0], [0.0,0.0], 0.0, 0.0,0.0, 0.0]
        current_pose2=Thi.get_pose(packed_pose) #packed_pose=>current_pose2
        result, out_image=Thi.inference(input_image,current_pose2) 
        out_image.show()
        
    #サンプル ３ inference(）を使用　Dict形式をget_pose_dicdでリポジトリの形式に変換　 イメージは毎回ロード
    if test=="3":
        input_image  = Image.open(filename)
        input_image.show()
        #サンプル Dict形式 
        #"mouth"には2種類の記述方法がある"lowered_corner"と”raised_corner”は左右がある
        #  "mouth":{"menue":"aaa","val":0.0},
        #  "mouth":{"menue":"lowered_corner","left":0.5,"right":0.0},　これはほとんど効果がない
        pose_dic={"eyebrow":{"menue":"happy","left":1.0,"right":0.0},
                "eye":{"menue":"wink","left":0.5,"right":0.0},
                "iris_small":{"left":0.0,"right":0.0},
                "iris_rotation":{"x":0.0,"y":0.0},
                "mouth":{"menue":"aaa","val":0.7},
                "head":{"x":0.0,"y":0.0},
                "neck":0.0,
                "body":{"y":0.0,"z":0.0},
                "breathing":0.0
                }
        pose=Thi.get_pose_dic(pose_dic)#Dic-> pose変換
        print(pose)
        result, out_image=Thi.inference(input_image,pose)
        out_image.show()
      
    #サンプル ４　inference_pos()を使用　パック形式　イメージは事前ロード
    if test=="4": 
        input_image = Image.open(filename)
        input_image.show()
        img_number=Thi.load_img(input_image,user_id)
        packed_pose=["happy", [0.5,0.0], "wink", [1.0,0.0], [0.0,0.0], [0.0,0.0], "ooo", [0.0,0.0], [0.0,0.0], 0.0, 0.0,0.0, 0.0]
        result, out_image=Thi.inference_pos(packed_pose,img_number,user_id) 
        out_image.show()
        
    #サンプル 5　inference_dic()を使用 　DICT形式で直接サーバを呼ぶ　イメージは事前ロード    
    if test=="5": 
        input_image = Image.open(filename)
        img_number=Thi.load_img(input_image,user_id)
        pose_dic=pose_dic_org #Pose 初期値
        current_pose_list=[]
        for i in range(20):
            start_time=time.time()
            current_pose_dic=pose_dic
            current_pose_dic["eye"]["menue"]="wink"#pose_dicに対して動かしたい必要な部分だけ操作できる
            current_pose_dic["eye"]["left"]=i*2/40 #
            start_time=time.time()
            result,out_image = Thi.inference_dic(current_pose_dic,img_number,user_id)
            image_show(out_image)
            print("Genaration time=",(time.time()-start_time)*1000,"mS")
            
    ##サンプル 6 inference_img() pose=リポジトリの形式(ベタ書き)　
    if test=="6": 
        input_image = Image.open(filename)
        input_image.show()
        img_number=Thi.load_img(input_image,user_id)
        for i in range(100):
            current_pose3= [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0, 0.0, i/100, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.78, 0.57, 0.63, 0.75, 0.49, 0.43,0.23, 0.65,1.0]
            result,out_image=Thi.inference_img(current_pose3,img_number,user_id)
            image_show(out_image)
       
            
    #サンプル 7　inference_img() poseはパック形式形式をリポジトリの形式に変換 イメージは事前ロード,パック形式で連続変化させる  
    if test=="7":
        mode="breastup"
        input_image = Image.open(filename)
        input_image.show()
        imge = np.array(input_image)
        imge = cv2.cvtColor(imge, cv2.COLOR_RGBA2BGRA)
        cv2.imshow("Loaded image",imge)
        cv2.waitKey()
        img_number=Thi.load_img(input_image,user_id)
        print("img_number=",img_number)
        for i in range(50):
            packed_current_pose=[
                "happy",[0.5,0.0],"wink", [i/50,0.0], [0.0,0.0], [0.0,0.0],"ooo", [0.0,0.0], [0.0,i*3/50],i*3/50, 0.0, 0.0, 0.0]
            start_time=time.time()
            current_pose=Thi.get_pose(packed_current_pose) #packed_pose=>current_pose2
            result,out_image=Thi.inference_img(current_pose,img_number,user_id)

            out_image = np.array(out_image)
            out_image = cv2.cvtColor(out_image, cv2.COLOR_RGBA2BGRA)
            image=upscale(url ,out_image, mode, scale=2)
            
            print("Genaration time=",(time.time()-start_time)*1000,"mS")
            cv2.imshow("Loaded image",image)
            cv2.waitKey(1)
            
        for i in range(100):
            packed_current_pose=[
                "happy", [0.5,0.0], "wink",[1-i/50,0.0], [0.0,0.0], [0.0,0.0], "ooo", [0.0,0.0], [0.0,3-i*3/50], 3-i*3/50, 0.0, 0.0, 0.0,]
            start_time=time.time()
            current_pose2=Thi.get_pose(packed_current_pose)#packed_current_pose==>current_pose2
            result,out_image=Thi.inference_img(current_pose2,img_number,user_id)

            out_image = np.array(out_image)
            out_image = cv2.cvtColor(out_image, cv2.COLOR_RGBA2BGRA)
            image=upscale(url ,out_image, mode, scale=2)
            
            print("Genaration time=",(time.time()-start_time)*1000,"mS")
            cv2.imshow("Loaded image",image)
            cv2.waitKey(1)
            
        for i in range(100):
            packed_current_pose=[
                "happy", [0.5,0.0], "wink", [i/100,i/100], [0.0,0.0], [0.0,0.0], "ooo", [0.0,0.0], [0.0,-3+i*3/50], -3+i*3/50,0.0, 0.0,0.0,]
            start_time=time.time()
            current_pose2=Thi.get_pose(packed_current_pose)#packed_current_pose==>current_pose2
            result,out_image=Thi.inference_img(current_pose2,img_number,user_id)

            out_image = np.array(out_image)
            out_image = cv2.cvtColor(out_image, cv2.COLOR_RGBA2BGRA)
            image=upscale(url ,out_image, mode, scale=2)
            
            print("Genaration time=",(time.time()-start_time)*1000,"mS")

            cv2.imshow("Loaded image",image)
            cv2.waitKey(1)
            
        for i in range(100):
            packed_current_pose=[
                "happy", [0.5,0.0], "wink", [0.0,0.0], [0.0,0.0], [0.0,0.0], "ooo",  [0.0,0.0], [0.0,3-i*3/100],  3-i*3/100,  0.0, 0.0, 0.0,]
            start_time=time.time()
            current_pose2=Thi.get_pose(packed_current_pose) #packed_current_pose==>current_pose2
            result,out_image=Thi.inference_img(current_pose2,img_number,user_id)

            out_image = np.array(out_image)
            out_image = cv2.cvtColor(out_image, cv2.COLOR_RGBA2BGRA)
            image=upscale(url ,out_image, mode, scale=2)

            print("Genaration time=",(time.time()-start_time)*1000,"mS")
            cv2.imshow("Loaded image",image)
            cv2.waitKey(1)
        cv2.imshow("Loaded image",image)
        cv2.waitKey(5000)
        

    #サンプル 8　inference_dic() 　poseはDICT形式で直接サーバを呼ぶ　イメージは事前ロード　  DICT形式で必要な部分のみ選んで連続変化させる
    if test=="8":
        mode="breastup"
        div_count=30
        input_image = Image.open(filename)
        imge = np.array(input_image)
        imge = cv2.cvtColor(imge, cv2.COLOR_RGBA2BGRA)
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
            start_time=time.time()
            result,out_image = Thi.inference_dic(current_pose_dic,img_number,user_id,"cv2")
            image=upscale(url ,out_image, mode, scale=2)
            print("Genaration time=",(time.time()-start_time)*1000,"mS")
            cv2.imshow("Loaded image",image)
            cv2.waitKey(1)
        for i in range(div_count):
            start_time=time.time()
            current_pose_dic["eye"]["left"]=1-i/(div_count/2)
            current_pose_dic["head"]["y"]=3-i*3/(div_count/2)
            current_pose_dic["neck"]=3-i*3/(div_count/2)
            current_pose_dic["body"]["y"]=5-i*5/(div_count/2)
            start_time=time.time()
            result,out_image = Thi.inference_dic(current_pose_dic,img_number,user_id,"cv2")
            image=upscale(url ,out_image, mode, scale=2)
            
            print("Genaration time=",(time.time()-start_time)*1000,"mS")

            cv2.imshow("Loaded image",image)
            cv2.waitKey(1)
        for i in range(div_count):
            start_time=time.time()
            current_pose_dic["eye"]["left"]=i/div_count
            current_pose_dic["eye"]["right"]=i/div_count
            current_pose_dic["head"]["y"]=-3+i*3/(div_count/2)
            current_pose_dic["neck"]=-3+i*3/(div_count/2)
            current_pose_dic["body"]["z"]=i*3/div_count
            current_pose_dic["body"]["y"]=-5+i*5/(div_count/2)
            start_time=time.time()
            result,out_image = Thi.inference_dic(current_pose_dic,img_number,user_id,"cv2")
            image=upscale(url ,out_image, mode, scale=2)
            
            print("Genaration time=",(time.time()-start_time)*1000,"mS")

            cv2.imshow("Loaded image",image)
            cv2.waitKey(1)
        for i in range(div_count):
            start_time=time.time()
            current_pose_dic["eye"]["left"]=0.0
            current_pose_dic["eye"]["right"]=0.0
            current_pose_dic["head"]["y"]=3-i*3/(div_count/2)
            current_pose_dic["neck"]=3-i*3/(div_count/2)
            current_pose_dic["body"]["z"]=3-i*3/div_count
            current_pose_dic["body"]["y"]=5-i*5/(div_count/2)
            start_time=time.time()
            result,out_image = Thi.inference_dic(current_pose_dic,img_number,user_id,"cv2")
            image=upscale(url ,out_image, mode, scale=2)
            
            print("Genaration time=",(time.time()-start_time)*1000,"mS")

            cv2.imshow("Loaded image",image)
            cv2.waitKey(1)
        cv2.imshow("Loaded image",image)
        cv2.waitKey(5000)

    #サンプル 9　inference_img() poseはパック形式形式をリポジトリの形式に変換 イメージは事前ロード,パック形式で連続変化させる  画像＝cv2形式
    if test=="9":
        mode="breastup"
        input_image = Image.open(filename)
        input_image.show()
        imge = np.array(input_image)
        imge = cv2.cvtColor(imge, cv2.COLOR_RGBA2BGRA)
        cv2.imshow("Loaded image",imge)
        cv2.waitKey()
        img_number=Thi.load_img(input_image,user_id)
        print("img_number=",img_number)
        for i in range(50):
            packed_current_pose=[
                "happy",[0.5,0.0],"wink", [i/50,0.0], [0.0,0.0], [0.0,0.0],"ooo", [0.0,0.0], [0.0,i*3/50],i*3/50, 0.0, 0.0, 0.0]
            start_time=time.time()
            current_pose=Thi.get_pose(packed_current_pose) #packed_pose=>current_pose2
            result,out_image=Thi.inference_img(current_pose,img_number,user_id,"cv2")

            out_image = np.array(out_image)
            out_image = cv2.cvtColor(out_image, cv2.COLOR_RGBA2BGRA)
            image=upscale(url ,out_image, mode, scale=2)
            
            print("Genaration time=",(time.time()-start_time)*1000,"mS")
            cv2.imshow("Loaded image",image)
            cv2.waitKey(1)
            
        for i in range(100):
            packed_current_pose=[
                "happy", [0.5,0.0], "wink",[1-i/50,0.0], [0.0,0.0], [0.0,0.0], "ooo", [0.0,0.0], [0.0,3-i*3/50], 3-i*3/50, 0.0, 0.0, 0.0,]
            start_time=time.time()
            current_pose2=Thi.get_pose(packed_current_pose)#packed_current_pose==>current_pose2
            result,out_image=Thi.inference_img(current_pose2,img_number,user_id,"cv2")

            out_image = np.array(out_image)
            out_image = cv2.cvtColor(out_image, cv2.COLOR_RGBA2BGRA)
            image=upscale(url ,out_image, mode, scale=2)
            
            print("Genaration time=",(time.time()-start_time)*1000,"mS")
            cv2.imshow("Loaded image",image)
            cv2.waitKey(1)
            
        for i in range(100):
            packed_current_pose=[
                "happy", [0.5,0.0], "wink", [i/100,i/100], [0.0,0.0], [0.0,0.0], "ooo", [0.0,0.0], [0.0,-3+i*3/50], -3+i*3/50,0.0, 0.0,0.0,]
            start_time=time.time()
            current_pose2=Thi.get_pose(packed_current_pose)#packed_current_pose==>current_pose2
            result,out_image=Thi.inference_img(current_pose2,img_number,user_id,"cv2")

            out_image = np.array(out_image)
            out_image = cv2.cvtColor(out_image, cv2.COLOR_RGBA2BGRA)
            image=upscale(url ,out_image, mode, scale=2)
            
            print("Genaration time=",(time.time()-start_time)*1000,"mS")

            cv2.imshow("Loaded image",image)
            cv2.waitKey(1)
            
        for i in range(100):
            packed_current_pose=[
                "happy", [0.5,0.0], "wink", [0.0,0.0], [0.0,0.0], [0.0,0.0], "ooo",  [0.0,0.0], [0.0,3-i*3/100],  3-i*3/100,  0.0, 0.0, 0.0,]
            start_time=time.time()
            current_pose2=Thi.get_pose(packed_current_pose) #packed_current_pose==>current_pose2
            result,out_image=Thi.inference_img(current_pose2,img_number,user_id,"cv2")

            out_image = np.array(out_image)
            out_image = cv2.cvtColor(out_image, cv2.COLOR_RGBA2BGRA)
            image=upscale(url ,out_image, mode, scale=2)

            print("Genaration time=",(time.time()-start_time)*1000,"mS")
            cv2.imshow("Loaded image",image)
            cv2.waitKey(1)
        cv2.imshow("Loaded image",image)
        cv2.waitKey(5000)
            
if __name__ == "__main__":
    main()
