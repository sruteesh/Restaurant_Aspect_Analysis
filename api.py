# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 11:49:45 2016

@author: drive
"""
import json

from flask import request, url_for
# import flask
from flask_api import FlaskAPI, status, exceptions
from flask import Flask
import pandas as pd
# import numpy as np
# from skimage import io
# import argparse,os
# import caffe
# import sys
# import init
# import cv2,skimage
# import time
# from werkzeug import secure_filename
#import this

from sentiment_neuron.encoder import Model
import aspects_api_v1.extract_aspects as aspects_model

###############################################################################
app = FlaskAPI(__name__)
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

###############################################################################

# def allowed_file(filename):
#     ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg'])
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

###############################################################################

# def logging():
    
#     user=str(request.data.get('user', ''))
#     key = str(request.data.get('key', ''))
    
#     print "#########################"
#     print ''
#     print 'IP: ',str(request.remote_addr)
#     print 'User: ',user
#     print 'Key: ',key
    
#     try:
#         url = str(request.data.get('url', ''))
#         print 'URL: ',url
#     except:
#         pass
    
#     print ''
    
#     with open("log.txt", "a") as myfile:
#         myfile.write("#########################\n")
#         myfile.write("\n")
#         myfile.write('IP: '+str(request.remote_addr)+'\n')
#         myfile.write('User: '+user+'\n')
#         myfile.write('Key: '+key+'\n')
#         try:
#             url = str(request.data.get('url', ''))
#             myfile.write('URL: '+url+'\n')
#         except:
#             myfile.write("\n")

###############################################################################

# def get_image():
    
#     url=''
    
#     i=None
    
#     if os.path.isfile('temp/temp.jpg'):
#         os.remove('temp/temp.jpg')
        
    
#     try:
#         url = str(request.data.get('url', ''))
        
#         try:
#             i=io.imread(url)
            
#             if i.shape[2]>3:
#                 return i[:,:,:3]
#             else:
#                 return i
#         except:
#             pass
                
#     except:
        
#         pass
        
#     try:
#         file = request.files['filedata']
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file.save(os.path.join('temp/', 'temp.jpg'))

            
#             try:
#                 i=io.imread('temp/temp.jpg')
#                 if i.shape[2]>3:
#                     return i[:,:,:3]
#                 else:
#                     return i
            
#             except:
#                 return None
#     except:
#            return i

###############################################################################

# def authenticate():
    
#     userdict={'test':'test1234', 'drive':'drive12345','demo':'13977f49-e095-4c3c-a9ce-8432f61096c8'}
    
#     user=str(request.data.get('user', ''))
#     key = str(request.data.get('key', ''))


#     if user not in userdict:
#         return 'Not authenticated'
            
#     if key!=userdict[user]:
        
#         return 'Not authenticated'
    
#     else:
#         return 'Authenticated'

###############################################################################
####    Predict scene
# @app.route("/scene/", methods=['POST','PUT'])

# def predict_scene():
    
#     global attr
#     logging()
#     auth=authenticate()
    
#     if auth=='Not authenticated':
#         return flask.jsonify(Message='Invalid user ID / Password !')
    
#     i=get_image()
    
#     if i==None:
#         return flask.jsonify(Message='Please check the image url / file !') 
   
    
#     res=attr.predict_place(i)


#     return flask.jsonify(Scene_1=res[0],Location_of_scene_1=res[1],Type_of_scene_1=res[2],Scene_2=res[3],Location_of_scene_2=res[4],Type_of_scene_2=res[5],
#                          Scene_3=res[6],Location_of_scene_3=res[7],Type_of_scene_3=res[8])
    
# ###############################################################################
# ####    Predict objects

# @app.route("/objects/", methods=['POST','PUT'])

# def predict_object():
    
#     global imagenet
#     logging()
#     auth=authenticate()
    
#     if auth=='Not authenticated':
#         return flask.jsonify(Message='Invalid user ID / Password !')
    
#     i=get_image()
    
#     if i==None:
#         return flask.jsonify(Message='Please check the image url / file !') 
   
    
#     res=imagenet.predict(i)
    
#     print res
    
#     return flask.jsonify(Object_1=res[0],Object_2=res[1],Object_3=res[2])

# ###############################################################################
# ####    Predict sports

# @app.route("/sports/", methods=['POST','PUT'])

# def predict_sports():
    
#     global sports
#     logging()
#     auth=authenticate()
    
#     if auth=='Not authenticated':
#         return flask.jsonify(Message='Invalid user ID / Password !')
    
#     i=get_image()
    
#     if i==None:
#         return flask.jsonify(Message='Please check the image url / file !') 
   
    
#     res=sports.predict_sport(i)
    
#     print res
    
#     return flask.jsonify(Sport_1=res[0],Sport_2=res[1],Sport_3=res[2])

# ###############################################################################
# ####    Predict fashion

# @app.route("/fashion/", methods=['POST','PUT'])

# def predict_fashion():
    
#     global fashion
#     logging()
#     auth=authenticate()
    
#     if auth=='Not authenticated':
#         return flask.jsonify(Message='Invalid user ID / Password !')
    
#     i=get_image()
    
#     if i==None:
#         return flask.jsonify(Message='Please check the image url / file !') 
   
    
#     res=fashion.predict_fashion(i)
    
#     return flask.jsonify(Fashion_1=res[0],Fashion_2=res[1],Fashion_3=res[2])

# ###############################################################################
# ####    Predict Furniture

# @app.route("/furniture/", methods=['POST','PUT'])

# def predict_furniture():
    
#     global furniture
#     logging()
#     auth=authenticate()
    
#     if auth=='Not authenticated':
#         return flask.jsonify(Message='Invalid user ID / Password !')
    
#     i=get_image()
    
#     if i==None:
#         return flask.jsonify(Message='Please check the image url / file !') 
   
    
#     res=furniture.predict_furniture(i)
    
#     return flask.jsonify(Furniture_1=res[0],Furniture_2=res[1],Furniture_3=res[2])
    
###############################################################################

@app.route("/sentiment/", methods=['POST','PUT'])

def get_sentiment():
    
    global neuron_model
    # logging()

    # return "Bye"
    # auth=authenticate()
    
    # if auth=='Not authenticated':
    #     return flask.jsonify(Message='Invalid user ID / Password !')
    
    text=request.json["text"]
    text_type=request.json["text_type"]

    text_features = neuron_model.transform(text)

    # return "hi"

    prediced = {}
    for no in range(len(text_features[0])):
        for i,res in enumerate(text_features):
            if no in prediced:
                prediced[no].append(res[no])
            else:
                prediced[no] = []
                prediced[no].append(res[no])
    # print (prediced[2388])
    return str(prediced[2388])+'\n'
    # i=get_image()
    
    # if i==None:
    #     return flask.jsonify(Message='Please check the image url / file !') 
   
    
    # res=furniture.predict_furniture(i)
    
    # return flask.jsonify(Sentiment=prediced[2388])
    
##############################################################################
@app.route("/aspects/", methods=['POST','PUT'])

def get_aspects():
    
    text=request.json["text"]
    text_type=request.json["text_type"]

    aspects_res = pd.DataFrame(aspects_model.get_aspects(text))
    print (aspects_res['review_sentence'])
    try:
        sent = predict_sentiment(aspects_res['review_sentence'])
        aspects_res['sentiment'] = sent
    except Exception as e:
        print (e)

    # return '\n' + json.dumps(text_features) + '\n'
    print (aspects_res)
    return '\n'+json.dumps(aspects_res.to_json()) + '\n'


def predict_sentiment(text):
    
    # global neuron_model
    # # logging()

    # # return "Bye"
    # # auth=authenticate()
    
    # # if auth=='Not authenticated':
    # #     return flask.jsonify(Message='Invalid user ID / Password !')
    
    # text=request.json["text"]
    # text_type=request.json["text_type"]

    text_features = neuron_model.transform(text)

    # return "hi"

    prediced = {}
    for no in range(len(text_features[0])):
        for i,res in enumerate(text_features):
            if no in prediced:
                prediced[no].append(res[no])
            else:
                prediced[no] = []
                prediced[no].append(res[no])
    # print (prediced[2388])
    return prediced[2388]

def main():

    global neuron_model
    neuron_model = Model()
    # caffe.set_mode_gpu()
    # caffe.set_device(0)
    
    
    
    # #####   Location of model files
    # p_root='files/places/'
    # attr=init.Attributes(p_root+'places.prototxt',p_root+'places.caffemodel',p_root+'synset.txt')
    
    # #### Imagenet
    
    # i_root='files/inception_resnet/'
    # imagenet=init.Imagenet(i_root+'inception_resnet_v2_2016_08_30.ckpt')
    
    # #### Sports
    
    # s_root='files/sports/'
    
    # sports=init.Sports(s_root+'deploy.prototxt',s_root+'sports.caffemodel',s_root+'synset.txt',s_root+'mean.binaryproto')
    
    # #### Fashion
    
    # f_root='files/fashion/'
    
    # fashion=init.Fashion(f_root+'deploy.prototxt',f_root+'fashion.caffemodel',f_root+'synset.txt',f_root+'mean.binaryproto')

    # #### Furniture
    
    # furn_root='files/furniture/'
    
    # furniture=init.Furniture(furn_root+'deploy.prototxt',furn_root+'furniture.caffemodel',furn_root+'synset.txt',furn_root+'mean.binaryproto')
    # text = ['demo!']
    # text_features = model.transform(text)

    app.run(host='192.168.0.102',port=8085)
    
if __name__=='__main__':
    main()