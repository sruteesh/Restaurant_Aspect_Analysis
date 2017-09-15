# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 11:49:45 2016

@author: drive
"""
import json

from flask import request, url_for
import flask
from flask_api import FlaskAPI, status, exceptions
from flask import Flask, jsonify
from flask_cors import CORS

import pandas as pd
import simplejson

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
from crossdomain import crossdomain
from sentiment_neuron.encoder import Model
import aspects_api_v1.extract_aspects_v2 as aspects_model


###############################################################################
# app = FlaskAPI(__name__)
app = Flask(__name__,
            static_url_path='', 
            static_folder='web/static',
            template_folder='web/templates')
CORS(app)

# @app.after_request
# def after_request(response):
#   response.headers.add('Access-Control-Allow-Origin', '*')
#   response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
#   response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
#   return response


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


@app.route("/textapi/", methods=['POST','PUT','GET'])
# @crossdomain(origin='*')
def get_textapi():
    global neuron_model
    # print ('Sentiment')
    # print (request.form)
    if request.method == 'GET':
    	return flask.render_template('index.php')
    elif request.method == 'POST':
        clientDetails()
        print(request.form)
        return endpointManager(request.form)
	    # if request.form["domain"].strip() == 'sentiment':
	    # 	return flask.json.dumps({"Sentiment ": predict_sentiment([request.form["text"]])})
	    # elif request.form["domain"].strip() in ['hotels','restaurants'] :
	    # 	return predict_aspects([request.form["text"]],request.form["domain"])
	    # else :
	    # 	return request.form["domain"] + "    Bad Request"
###############################################################################

@app.route("/sentiment/", methods=['POST','PUT'])
# @crossdomain(origin='*')

def get_sentiment():
    
    global neuron_model
    print ('Sentiment')
    print (request.form)
    # if request.method == 'GET':
    # 	return flask.render_template('index.php')
    # elif request.method == 'POST':
	   #  if request.form["domain"].strip() == 'sentiment':
	   #  	return str(predict_sentiment([request.form["text"]]))
	   #  elif request.form["domain"].strip() in ['hotels','restaurants'] :
	   #  	return predict_aspects([request.form["text"]],request.form["domain"])
	   #  else :
	   #  	return request.form["domain"] + "    Bad Request"

    # logging()
    # return "Bye"
    # auth=authenticate()
    
    # if auth=='Not authenticated':
    #     return flask.jsonify(Message='Invalid user ID / Password !')
    # print (type(request))
    # print (request.form)
    # # print j
    # return str({'resul': 'sucesseeeeeee'})
    # # print (request.json)
    text=request.json["text"]
    return predict_sentiment(text)
    # domain=request.json["domain"]	
    # text_features = neuron_model.transform([text])

    # # return "hi"

    # prediced = {}
    # for no in range(len(text_features[0])):
    #     for i,res in enumerate(text_features):
    #         if no in prediced:
    #             prediced[no].append(res[no])
    #         else:
    #             prediced[no] = []
    #             prediced[no].append(res[no])
    # # print (prediced[2388])
    # return str(prediced[2388])+'\n'
    # # i=get_image()
    
    # if i==None:
    #     return flask.jsonify(Message='Please check the image url / file !') 
   
    
    # res=furniture.predict_furniture(i)
    
    # return flask.jsonify(Sentiment=prediced[2388])
    
##############################################################################
@app.route("/aspects/", methods=['POST','PUT'])

def get_aspects():
    
    text=request.json["text"]
    domain=request.json["domain"]

    aspects_res = aspects_model.get_aspects(text,domain=domain)
    final_aspect_res=[]
    with open('sample_results.json','w') as fin:
        for _,rev in enumerate(aspects_res):
            sentences=[]
            for sent in rev:
                try:
                    senti = predict_sentiment([sent['review_sentence']])
                    sent['sentiment'] = round(float(senti),3)
                    sent['review_id'] = _
                    sentences.append(sent)
                except Exception as e:
                    print (e)
            simplejson.dump(sentences,fin)
            fin.write('\n')
            final_aspect_res.append(sentences)
    # return '\n' + json.dumps(text_features) + '\n'
    return '\n'+str(final_aspect_res) + '\n'


def predict_sentiment(text):

    text_features = neuron_model.transform(text)
    prediced = {}
    for no in range(len(text_features[0])):
        for i,res in enumerate(text_features):
            if no in prediced:
                prediced[no].append(res[no])
            else:
                prediced[no] = []
                prediced[no].append(res[no])
    return prediced[2388][0]

def predict_aspects(text,domain):
    aspects_res = aspects_model.get_aspects(text,domain=domain)
    final_aspect_res=[]
    for _,rev in enumerate(aspects_res):
        sentences=[]
        for sent in rev:
            try:
                senti = predict_sentiment([sent['review_sentence']])
                sent['sentiment'] = senti
                sent['review_id'] = _
                sentences.append(sent)
            except Exception as e:
                print (e)
        final_aspect_res.append(sentences)
    print (flask.json.dumps(final_aspect_res))  
    return flask.json.dumps(final_aspect_res)

def clientDetails():
    # print ("Client Details")
    # print ('ip: ' + req.remote_addr)
    # print (dir(req))
    print ('ip: ' + request.remote_addr)

def endpointManager(data):
    if data["domain"].strip() == 'sentiment':
        print(data  )
        return flask.json.dumps({"Sentiment ": predict_sentiment([data["text"]])})
    elif data["domain"].strip() in ['hotels','restaurants'] :
        return predict_aspects([data["text"]],data["domain"])
    else :
        return data["domain"] + "    Bad Request"

def main():

    global neuron_model
    neuron_model = Model()
    app.run(host='localhost',port=8085)
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


    
if __name__=='__main__':
    main()