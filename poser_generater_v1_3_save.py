import numpy as np
import cv2
from PIL import Image
import argparse
from time import sleep
import time
import multiprocessing
from multiprocessing import Process, Value, Manager
import ctypes
from poser_client_tkhmp_upmp_v1_3_class import TalkingHeadAnimefaceInterface
from tkh_up_scale import upscale

#PIL形式の画像を動画として表示
def image_show(imge):
    imge = np.array(imge)
    imge = cv2.cvtColor(imge, cv2.COLOR_RGBA2BGRA)
    cv2.imshow("Loaded image",imge)
    cv2.waitKey(1)
    
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
        #                eye_brow             eye       iris_small  iris_lotat       mouth       head     neck    body   breath
        #                 L,  R              L    R       L   R      L   R           L    R      X   Y     Z     Y    Z
        #          0      1    2      3     4[0] 4[1]   5[0] 5[1]   6[0] 6[1]  7    8[0] 8[1]   9[0] 9[1]  10   11   12   13
        # pose=["happy",[0.0,0.0],  "wink", [0.0,0.0],  [0.0,0.0], [0.0,0.0],"ooo", [0.0,0.0], [0.0,0.0], 0.0,  0.0, 0.0, 0.0]

class TalkingHeadAnimefaceGenerater():
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
    current_pose_dic = Manager().dict(pose_dic_org)
        
    def __init__(self,Thi,img_number,user_id,mode,scale,fps,pose_dic_org):
        self.img_number=img_number
        self.user_id=user_id
        self.mode=mode
        self.scale=scale
        self.fps=fps


        self.pose_dic_org={"eyebrow":{"menue":"happy","left":0.0,"right":0.0},
                  "eye":{"menue":"wink","left":0.0,"right":0.0},
                  "iris_small":{"left":0.0,"right":0.0},
                  "iris_rotation":{"x":0.0,"y":0.0},
                  "mouth":{"menue":"aaa","val":0.0},
                  "head":{"x":0.0,"y":0.0},
                  "neck":0.0,
                  "body":{"y":0.0,"z":0.0},
                  "breathing":0.0,
                  }
        
        #Thiの初期化
        self.Thi=Thi

        #init_dic=Thi.get_init_dic()


        self.current_pose_dic = Manager().dict(pose_dic_org)

        self.queue_in_pose = None
        self.queue_out_image = None

        self.emotion_update=False     #喜怒哀楽の顔
        self.wink_update   =False     #ウインク、両目の瞬き
        self.mouse_update  =False     #リップシンク
        self.head_neck_update =False  #頭を動かす
        self.body_update   =False     #体を動かす

        self.req_all=Value("i",0)     #全Pose_Dicで動かす
        

        self.out_image = np.zeros((512, 512, 3), dtype=np.uint8)#upscaleが最初に呼び出される時に画像ができていないので初期値を設定
        self.global_out_image=self.out_image


    #start process
    def start_mp_generatae_process(self,fps):
        self.q_in_wink = multiprocessing.Queue()  # 入力Queueを作成
        self.q_in_wink = multiprocessing.Queue()  # 入力Queueを作成
        self.queue_out_image = multiprocessing.Queue()  # 出力Queueを作成
        #self.mp_generatae_proc = multiprocessing.Process(target=self._mp_generatae, args=(self.Thi,self.global_out_image,self.current_pose_dic,self.queue_out_image,self.img_number,self.user_id,self.mode,self.scale,self.req_all))  #process作成
        self.mp_generatae_proc = multiprocessing.Process(target=self._mp_generatae, args=(self.Thi,
                                                                                          self.global_out_image,
                                                                                          TalkingHeadAnimefaceGenerater.current_pose_dic,
                                                                                          self.queue_out_image,
                                                                                          self.img_number,
                                                                                          self.user_id,
                                                                                          self.mode,
                                                                                          self.scale,))  #process作成
        self.mp_generatae_proc.start() #process開始
        #self.mp_generatae_proc.join()

    """  
    def start_mp_generatae_process(self,fps):
        print("start_mp_generatae_process")
        self.queue_out_image = multiprocessing.Queue()  # 出力Queueを作成
        mp_generatae_proc = multiprocessing.Process(target=self._mp_generatae_t,args=(self.global_out_image,
                                                                                      self.queue_out_image,
                                                                                      self.img_number,
                                                                                      self.user_id,
                                                                                      self.mode,
                                                                                      self.scale
                                                                                      ))
        mp_generatae_proc.start() #process開始
                          

    def _mp_generatae_t(self,global_out_image,queue_out_image,img_number,user_id,mode,scale):
    #def _mp_generatae_t(self):
        print("_mp_generatae_t started")
        while True:
            if self.req_all.value==1:
                print("@@@@@@@@@@@@ req_all",self.req_all.value)
                self.req_all.value=0
            sleep(0.01)
    """

    #mp_generatae_processプロセス停止関数--terminate
    def mp_generatae_process_terminate(self):
        while not self.queue_out_image.empty():
            self.queue_out_image.get_nowait()
        self.mp_generatae_proc.terminate()#サブプロセスの終了
        print("mp_generatae process terminated")
        
    #_mp_generatae(): 部位別POSEリクエストを確認してリクエストがあれば各部のPOSEを反映する
    #def _mp_generatae(self, Thi, global_out_image , current_pose_dic , queue_out_image , img_number , user_id , mode , scale , req_all):
    def _mp_generatae(self, Thi, global_out_image , current_pose_dict , queue_out_image , img_number , user_id , mode , scale):
        print("+++++ Start mp_generatae process")
        while True:
            #if self.queue_in_pose.empty()==False:
            #    self.current_pose_dic=self.queue_in_pose.get()
            #print("@@@@@@@@@@@@ req_all",self.req_all)
            start_time=time.time()
            pose_request=False
            """
            if self.emotion_update:
                self.emotion_pdate=False
                pose_request=True
            elif self.wink_update:
                self.wink_update=False
                pose_request=True
            elif self.mouse_update:
                self_mouse_update=False
                pose_request=True
            elif self.head_neck_update:
                self.head_neck_update=False
                pose_request=True
            elif self.body_update:
                self.body_update=False
                pose_request=True
            """
            print("self.req_all.value=",self.req_all.value)
            if self.req_all.value==1:
                print("@@@@@@@@@@@@ req_all",self.req_all.value)
                pose_request=True
                #self.req_all=False
                
            if pose_request:
                print("==========- self.req_all=True",self.req_all.value)
                
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
                
                #print("XXXXXXXX current_pose_dic", current_pose_dict)
                print("YYYYYYYYYYYY pose_dic_org", pose_dic_org)
                
                self.out_image=Thi.mp_dic2image_frame(global_out_image, pose_dic_org ,img_number, user_id, mode, scale, 0)
                if self.queue_out_image.empty():
                    self.queue_out_image.put(self.out_image)
                self.req_all.value=0
                print("==========- self.req_all=True",self.req_all.value)
                
            #print("mp_generatae 1/fps - (time.time()-start_time)=",1/self.fps - (time.time()-start_time))
            if (1/self.fps - (time.time()-start_time))>0:
                sleep(1/self.fps - (time.time()-start_time))
            else:
                print("mp_generatae Remain time is minus")

            sleep(0.02)
            
    #stop_process

    def pose_wink(self,l_r, time, fps):
        pose_data_count=int(time/fps)/2
        current_pose_dic["eye"]["menue"]="wink"
        if l_r=="l":
            pose_step = (1.0 - self.current_pose_dic["eye"]["left"])/pose_data_count
            for _ in range(pose_data_count):
                self.current_pose_dic["eye"]["left"] += pose_step
            for _ in range(pose_data_count):
                self.current_pose_dic["eye"]["left"] -= pose_step
            self.eye_update=1
        elif l_r=="r":
            for _ in range(pose_data_count):
                self.current_pose_dic["eye"]["right"] += pose_step
            for _ in range(pose_data_count):
                self.current_pose_dic["eye"]["right"] -= pose_step
            self.eye_update=1
        elif l_r=="b":
            for _ in range(pose_data_count):
                self.current_pose_dic["eye"]["left"] += pose_step
                self.current_pose_dic["eye"]["right"] += pose_step
            for _ in range(pose_data_count):
                self.current_pose_dic["eye"]["left"] -= pose_step
                self.current_pose_dic["eye"]["right"] -= pose_step
            self.eye_update=1
        else:
            print("Eye error")
            self.eye_update=0

    def pose_all(self,pose_dic):
        print("++++++++++++ pose_all- self.req_all=",self.req_all.value)
        print("~~~~~~~~~~~~ pose_all")
        self.req_all.value= 1
        print("~~~~~~~~~~~~ pose_all- self.req_all=",self.req_all.value)
        TalkingHeadAnimefaceGenerater.current_pose_dic=pose_dic
        #print(pose_dic)
        print("Pose_all=",TalkingHeadAnimefaceGenerater.current_pose_dic)
        #if self.queue_in_pose.empty():
        #    self.queue_in_pose.put(self.current_pose_dic)
        
        if self.queue_out_image.empty()==False:
            self.out_image = self.queue_out_image.get()
            
        return self.out_image
        
def main():
    parser = argparse.ArgumentParser(description='Talking Head')
    parser.add_argument('--filename','-i', default='000002.png', type=str)
    parser.add_argument('--test', default=0, type=int)
    parser.add_argument('--host', default='http://0.0.0.0:8001', type=str)
    parser.add_argument('--esr', default='http://0.0.0.0:8008', type=str)
    args = parser.parse_args()
    test =args.test
    filename =args.filename

    user_id=0 #便宜上設定している。0~20の範囲。必ず設定すること
    
    tkh_url=args.host
    esr_url=args.esr + "/resr_upscal/"

    #Thiの初期化
    Thi=TalkingHeadAnimefaceInterface(tkh_url)  # tkhのホスト
                                                # アップスケールのURLはプロセスで指定
    #pose_dic_orgの設定。サーバからもらう
    pose_dic_org = Thi.get_init_dic()

    #アップスケールとtkhプロセスの開始
    Thi.create_mp_upscale(esr_url)
    Thi.create_mp_tkh()

    

    #サンプル 1　inference_dic() 　poseはDICT形式で直接サーバを呼ぶ　イメージは事前ロード　  DICT形式で必要な部分のみ選んで連続変化させる
    if test==1:
        fps=20
        #mode="breastup"  #  "breastup" , "waistup" , upperbody" , "full"
        mode=[55,155,200,202] #[top,left,hight,whith]
        #mode="waistup"
        scale=2 # 2/4/8
        div_count=10
        input_image = Image.open(filename)
        imge = np.array(input_image)
        imge = cv2.cvtColor(imge, cv2.COLOR_RGBA2BGRA)
        result_out_image = imge
        cv2.imshow("image",imge)
        cv2.waitKey() #ここで一旦止まり、キー入力で再開する
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

        

        Tkg=TalkingHeadAnimefaceGenerater(Thi,img_number,user_id,mode,scale,fps,pose_dic_org)
        print("#kuru!!!")
        Tkg.start_mp_generatae_process(fps)
        print("#konai!!!")
        #sleep(0.5)
        
        pose_dic=pose_dic_org #Pose 初期値
        current_pose_list=[]
        for i in range(int(div_count/2)):
            start_time=time.time()
            current_pose_dic=pose_dic
            #current_pose_dic["eye"]["menue"]="wink"
            #current_pose_dic["eye"]["left"]=i/(div_count/2)
            current_pose_dic["head"]["y"]=i*3/(div_count/2)
            current_pose_dic["neck"]=i*3/(div_count/2)
            current_pose_dic["body"]["y"]=i*5/(div_count/2)
            

            
            #current_pose_dic = pose_dic_org
            
            result_out_image = Tkg.pose_all(current_pose_dic)
            
            #result_out_image = Thi.mp_dic2image_frame(result_out_image,current_pose_dic,img_number,user_id,mode,scale,fps=0)
            cv2.imshow("Loaded image",result_out_image)
            cv2.waitKey(1)
            print("1/fps - (time.time()-start_time)=",1/fps - (time.time()-start_time))
            if (1/fps - (time.time()-start_time))>0:
                sleep(1/fps - (time.time()-start_time))
            else:
                print("Remain time is minus")
            print("Genaration time=",(time.time()-start_time)*1000,"mS")
        """
        for i in range(div_count):
            start_time=time.time()
            current_pose_dic["eye"]["left"]=1-i/(div_count/2)
            current_pose_dic["head"]["y"]=3-i*3/(div_count/2)
            current_pose_dic["neck"]=3-i*3/(div_count/2)
            current_pose_dic["body"]["y"]=5-i*5/(div_count/2)
            result_out_image = Thi.mp_dic2image_frame(result_out_image,current_pose_dic,img_number,user_id,mode,scale,fps=0)
            cv2.imshow("Loaded image",result_out_image)
            cv2.waitKey(1)
            print("1/fps - (time.time()-start_time)=",1/fps - (time.time()-start_time))
            if (1/fps - (time.time()-start_time))>0:
                sleep(1/fps - (time.time()-start_time))
            else:
                print("Remain time is minus")
            print("Genaration time=",(time.time()-start_time)*1000,"mS")
        for i in range(div_count):
            start_time=time.time()
            current_pose_dic["eye"]["left"]=i/div_count
            current_pose_dic["eye"]["right"]=i/div_count
            current_pose_dic["head"]["y"]=-3+i*3/(div_count/2)
            current_pose_dic["neck"]=-3+i*3/(div_count/2)
            current_pose_dic["body"]["z"]=i*3/div_count
            current_pose_dic["body"]["y"]=-5+i*5/(div_count/2)
            result_out_image = Thi.mp_dic2image_frame(result_out_image,current_pose_dic,img_number,user_id,mode,scale,fps=0)
            cv2.imshow("Loaded image",result_out_image)
            cv2.waitKey(1)
            print("1/fps - (time.time()-start_time)=",1/fps - (time.time()-start_time))
            if (1/fps - (time.time()-start_time))>0:
                sleep(1/fps - (time.time()-start_time))
            else:
                print("Remain time is minus")
            print("Genaration time=",(time.time()-start_time)*1000,"mS")
        for i in range(div_count):
            start_time=time.time()
            current_pose_dic["eye"]["left"]=0.0
            current_pose_dic["eye"]["right"]=0.0
            current_pose_dic["head"]["y"]=3-i*3/(div_count/2)
            current_pose_dic["neck"]=3-i*3/(div_count/2)
            current_pose_dic["body"]["z"]=3-i*3/div_count
            current_pose_dic["body"]["y"]=5-i*5/(div_count/2)
            result_out_image = Thi.mp_dic2image_frame(result_out_image,current_pose_dic,img_number,user_id,mode,scale,fps=0)
            cv2.imshow("Loaded image",result_out_image)
            cv2.waitKey(1)
            print("1/fps - (time.time()-start_time)=",1/fps - (time.time()-start_time))
            if (1/fps - (time.time()-start_time))>0:
                sleep(1/fps - (time.time()-start_time))
            else:
                print("Remain time is minus")
            print("Genaration time=",(time.time()-start_time)*1000,"mS")
        """
        cv2.imshow("Loaded image",result_out_image)
        cv2.waitKey(5)
        cv2.waitKey(1000)
    #サブプロセスの終了
    Thi.up_scale_proc_terminate()
    Thi.tkh_proc_terminate()

    Tkg.mp_generatae_process_terminate()
    print("end of test")
            
if __name__ == "__main__":
    main()
