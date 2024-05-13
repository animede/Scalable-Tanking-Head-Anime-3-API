import time
import os
import sys
import numpy as np
from PIL import Image
import torch
from datetime import datetime
import requests

import shutil
import glob
import socket
import pickle
import argparse
import cv2

from  face_d_api_class_client import AnimeFace_det
from tkh_up_scale import upscale

global fd_url
global ud_url

def main():
    global fd_url
    global up_url
    parser = argparse.ArgumentParser(description='Talking head')
    parser.add_argument("--fd_url",   type=str,  default="http://0.0.0.0:50001",  help="サービスを提供するip アドレスを指定。")
    parser.add_argument("--up_url",   type=str,  default="http://0.0.0.0:8008/resr_upscal/", help="サービスを提供するポートを指定。")
    parser.add_argument('--test', default=False, type=bool)
    parser.add_argument('--filename','-i', default='img_gen7E3ufz-3.png', type=str)
    parser.add_argument('--shape','-s', default='face', type=str)

    args = parser.parse_args()
    test  = args.test
    print("test=",test)
    filename =args.filename
    img_shape =args.shape
    fd_url= args.fd_url
    up_url= args.up_url    
    print("fd_url=",fd_url)
    print("up_url=",up_url)

                
    #任意の入力画像からTalking-Head用画像を生成、remrgb使用、白黒2値のマスクを用いた改良が必要、haikeiバージョンも必要
    if test==True:
        print("test7")
        pil_input_image = Image.open(filename)
        
        result , pil_w_img = image_data_form(pil_input_image,"pil")

        pil_w_img.show()

# *********************** 　汎用背景削除　from del_bkg_api  import del_bkg_out 
def del_bkg_out(img , img_mode, url="http://0.0.0.0:8007/del_bkg/"):
    if img_mode=="pil": #pilの場合はcvに変換
        img= np.array( img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #カラーチャンネル変換
   # 以下cv/pil共通      バイナリデータをPOSTリクエストで送信
    _, img_encoded = cv2.imencode('.jpg', img)
    response = requests.post(url, files={"file": ("image.jpg", img_encoded.tobytes(), "image/jpeg"),"mode":(None,img_mode)})
    all_data =response.content
    frame_data = (pickle.loads(all_data))#元の形式にpickle.loadsで復元 #形式はimg_mode指定の通り
    return frame_data #形式はimg_mode指定の通り
       
# *************** TKH用にinput_imageをフォーミングすす。out_formで出力をpilかcvに指定、mask：True= マスク生成,del_flegで背景削除の有無を指定する。
#    例）Talking-Head用アライメント画像作成
#        input_image =image_data_form(input_image ,"pil",False)

def image_data_form(input_image,out_form="pil",mask=False,del_flag=True): #入出力の画像形式はout_formで指定
    global fd_url
    global up_url
    AF=AnimeFace_det(fd_url)

    result =True
    if out_form=="cv":#input is cv
        height, width, channels = input_image.shape
        if channels != 4: #αチャンネルがなければ背景を削除してチャンネル追加
            input_image = del_bkg_out(nput_image , "cv") #input_image = 背景を削除 OpeCV
    else: #input is pil
        if input_image.mode != "RGBA": #αチャンネルがなければ背景を削除してチャンネル追加
            input_image = del_bkg_out(input_image, "pil") #背景を削除
        np_w_img = np.array(input_image, dtype=np.uint8)
        input_image = cv2.cvtColor(np_w_img, cv2.COLOR_RGBA2BGRA) #input_image = 背景を削除 OpeCV
    cv_w_img = cv2.cvtColor(input_image, cv2.COLOR_BGRA2BGR)#Face detectのためにαチャンネルを削除
    imge, dnum, predict_bbox, pre_dict_label_index, scores =AF.face_det_head(cv_w_img)#face-head検出   
    print("dnum=",dnum,"bbox=",predict_bbox,"label=",pre_dict_label_index,"score=",scores)
    try:
            box=predict_bbox[0]
            #print("box= ",box)
            #face-head検出のバウンディングbox付きの大きさを元に画像の拡大率を計算（THKのフォフォームに合わせるため：Head=128標準 ）
            box_disp=(box[0],box[1]),(box[2],box[3])
            print(box_disp)
            #print("BOX SIZE=",int(box[0]-box[2]),int(box[1]-box[3]))
            resize_facter=128/int(box[0]-box[2])
            print("resize_facter=",resize_facter) #HeadからResizeのファクタを計算
    except:
            result = "resize error" #resize_facteの計算が正しく行えなかった場合は1=なにもしない。
            return result
        
    if resize_facter > 4:   #2倍以上の拡大は推奨できないのでエラー
        print("image is too small")
        result="image is too small"
    elif resize_facter > 2: #4倍して所定のサイズに縮小する
            input_image = upscale(up_url ,input_image, "full", 4) #upscale
            image_resaize=resize_facter/4
    elif resize_facter > 1: #2倍して所定のサイズに縮小する
            input_image = upscale(up_url ,input_image, "full", 2) #upscale
            image_resaize=resize_facter/2
    else: # 1> reasize >0 なのでそのまま縮小率として使う
        image_resaize=resize_facter
        
    height, width, channels = input_image.shape
    cv_resize_img = cv2.resize(input_image, dsize=(round(width*image_resaize),round(height*(image_resaize))),interpolation = cv2.INTER_AREA)
    height, width, channels = cv_resize_img.shape#縮小した画像のH,W取得
    print("resize_image h= ",height,"w= ",width,"Channels= ",channels)
    #バウンディングboxは検出字のresize_facterを使う
    top=int(box[3]*resize_facter)
    left=int(box[2]*resize_facter)
    bottom=int(box[1]*resize_facter)
    right=int(box[0]*resize_facter)    
    print("top=",top,"left=",left,"bottom=",bottom,"right=",right)  
    #print("Resize_BOX SIZE=",int(resize_facter*(box[0]-box[2])),int(resize_facter*(box[1]-box[3])))
    #cv2.imshow("cv_resize_img",cv_resize_img)
    #cv2.waitKey()

    #αチャンネル付き入力画像をPILイメージに変換
    nd_input_image = cv2.cvtColor(cv_resize_img, cv2.COLOR_BGRA2RGBA)
    pil_input_image = Image.fromarray(nd_input_image)
    # 512x512ピクセルで、全てのピクセルが透明な画像を作成
    pil_out_image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    # ペーストする位置を指定
    p_top = 64-top #バウンディングboxの位置が64pixよりも大きければ差がペースト位置、小さければ差がマイナスになっているがpilではOK
    p_left =192-left 
    # 画像の大きさを調整した入力画像をアルファチャンネルを考慮して前景画像を背景画像にペースト
    pil_out_image.paste(pil_input_image, (p_left, p_top), pil_input_image)
    #pil_out_image.show()
    if out_form=="pil":
        return result, pil_out_image
    elif out_form=="cv":
        np_w_img = np.array(pil_out_image, dtype=np.uint8)
        cv_out_image = cv2.cvtColor(np_w_img, cv2.COLOR_RGBA2BGRA) #input_image = 背景を削除 OpeCV
        return result, cv_out_image

#sr を使う　デフォルト　中速高精細
def resize(cv_imge,img_shape):        
            #OUTサイズが512のとき
            print(">>>>>------------------------->>>------------------------------->>Shape=",img_shape)
            if img_shape=="face_fine":
                cv_imge = cv_imge[64 :240-8 ,174+4: 350-4]
                cv_imge=sr(cv_imge,2) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                cv_imge=cv2.resize(cv_imge,dsize=(512,512),interpolation = cv2.INTER_AREA)
            elif img_shape=="face_variety":
                #cv_imge = cv_imge[48 :208 ,176: 338]
                cv_imge = cv_imge[64 :240-12 ,174+6: 350-6]
                cv_imge=sr(cv_imge,2) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<    
                cv_imge=cv2.resize(cv_imge,dsize=(512,512),interpolation = cv2.INTER_AREA)
            elif img_shape=="face_bkgrand":
                #cv_imge = cv_imge[48 :208 ,176: 338]
                cv_imge = cv_imge[64 :240-16 ,174+8: 350-8]
                cv_imge=sr(cv_imge,2) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<    
                cv_imge=cv2.resize(cv_imge,dsize=(512,512),interpolation = cv2.INTER_AREA)
            elif img_shape=="face_men":
                cv_imge = cv_imge[48 :208 ,176: 338]
                cv_imge=sr(cv_imge,2) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<    
                cv_imge=cv2.resize(cv_imge,dsize=(512,512),interpolation = cv2.INTER_AREA)
            elif img_shape=="chest_up":
                cv_imge = cv_imge[64 :256+16,160-8:352+8]            
                cv_imge=sr(cv_imge,2) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<    
                cv_imge=cv2.resize(cv_imge,dsize=(512,512),interpolation = cv2.INTER_AREA)
            elif img_shape=="west_up":
                cv_imge = cv_imge[48 :304-10 ,128+5: 384-5]
                cv_imge=sr(cv_imge,2) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<    
                cv_imge=cv2.resize(cv_imge,dsize=(512,512),interpolation = cv2.INTER_AREA)
            elif img_shape=="nee_up":
                cv_imge = cv_imge[48 :512 ,128+5: 384-5]
                cv_imge=sr(cv_imge,2) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<    
                cv_imge=cv2.resize(cv_imge,dsize=(512,512),interpolation = cv2.INTER_AREA)
            elif img_shape=="full_body":
               #cv_imge = cv_imge[32 :512 ,16: 496]
               cv_imge = cv_imge[0 :512,0: 512]
            else: #face
                cv_imge = cv_imge[48 :208 ,176: 338]
                cv_imge=sr(cv_imge,2) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<    
                cv_imge=cv2.resize(cv_imge,dsize=(512,512),interpolation = cv2.INTER_AREA)
            return cv_imge

def image_cut_512(cv_imge,img_shape):        
            #OUTサイズが512のとき
            if img_shape=="face_fine":
                cv_imge = cv_imge[48 :208 ,176: 338]
                cv_imge=cv2.resize(cv_imge,dsize=(512,512),interpolation = cv2.INTER_AREA)
            elif img_shape=="face_variety":
                cv_imge = cv_imge[64 :240-12 ,174+6: 350-6]
                cv_imge=cv2.resize(cv_imge,dsize=(512,512),interpolation = cv2.INTER_AREA)
            elif img_shape=="face_bkgrand":
                cv_imge = cv_imge[64 :240-8 ,174+4: 350-4]
                cv_imge=cv2.resize(cv_imge,dsize=(512,512),interpolation = cv2.INTER_AREA)
            elif img_shape=="face_men":
                cv_imge = cv_imge[48 :208 ,176: 338]
                cv_imge=cv2.resize(cv_imge,dsize=(512,512),interpolation = cv2.INTER_AREA)               
            elif img_shape=="chest_up":
                cv_imge = cv_imge[64 :256+16,160-8:352+8]           
                cv_imge=cv2.resize(cv_imge,dsize=(512,512),interpolation = cv2.INTER_AREA)
                
            elif img_shape=="nee_up":
                cv_imge = cv_imge[48 :304 ,128: 384]
                cv_imge=cv2.resize(cv_imge,dsize=(512,512),interpolation = cv2.INTER_AREA)
                
            elif img_shape=="west_up":
                cv_imge = cv_imge[48 :304 ,128: 384]
                cv_imge=cv2.resize(cv_imge,dsize=(512,512),interpolation = cv2.INTER_AREA)
                
            elif img_shape=="full_body":
               cv_imge = cv_imge[0 :512,0: 512]
            return cv_imge
        
#拡大をしない　高速
def image_cut(cv_imge,img_shape):        
            #OUTサイズが512のとき
            if img_shape=="face_fine":
                cv_imge = cv_imge[48 :208 ,176: 338]
            if img_shape=="chest_up":
                cv_imge = cv_imge[64 :256 ,160:352]            
            elif img_shape=="west_up":
                cv_imge = cv_imge[48 :304 ,128: 384]
            elif img_shape=="full_body":
               #cv_imge = cv_imge[32 :512 ,16: 496]
               cv_imge = cv_imge[0 :512,0: 512]
            return cv_imge


if __name__ == "__main__":
    main()
