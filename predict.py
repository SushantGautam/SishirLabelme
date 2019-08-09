import time

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

import os
from keras.preprocessing import image
import simplejson as json

import argparse

import numpy as np
import tensorflow as tf

import glob

BASE_PROJECT_DIR = "Images"
BASE_PREDICTIONS_DIR = "predictions"




app = Flask(__name__)

 # sudo fuser -k 5000/tcp & python3 flaskai.py


def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph




def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.compat.v1.Session()
  result = sess.run(normalized)

  return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label




model_file = \
    "prediction_model/output_graph.pb"
label_file = "prediction_model/output_labels.txt"
input_height = 299
input_width = 299
input_mean = 0
input_std = 255
input_layer = "Placeholder"
output_layer = "final_result"

graph = load_graph(model_file)

input_name = "import/" + input_layer
output_name = "import/" + output_layer
input_operation = graph.get_operation_by_name(input_name)
output_operation = graph.get_operation_by_name(output_name)
sess =  tf.compat.v1.Session(graph=graph)

@app.route('/')
def upload_files():
   return "use api"

@app.route('/predictapi/projects/<projectname>', methods=['GET', 'POST'])
def upload_file(projectname):
  projectpath = os.path.join(BASE_PROJECT_DIR, projectname)
  list_img_files = glob.glob(projectpath+'/*.jpg')+glob.glob(projectpath+'/*.png')+glob.glob(projectpath+'/*.jpeg')
  
  print(list_img_files)
  for file_name in list_img_files:
    
    
    # return file_name
    t = read_tensor_from_image_file(
        file_name,
        input_height=input_height,
        input_width=input_width,
        input_mean=input_mean,
        input_std=input_std)


    results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: t
        })
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)
    jsondata = '{'
    for i in top_k:
        # print (labels[i], results[i])
        jsondata+='"'+ str(labels[i])+ '": "'+str(results[i])+'", '
    jsondata+= '}'
    try:
      os.makedirs(os.path.join(BASE_PREDICTIONS_DIR,projectname))
    except OSError:
      print(OSError)
    else:
      print("succesful")
    file_name = file_name.replace(os.path.join(BASE_PROJECT_DIR, projectname)+'/','')
    print(file_name)
    with open(os.path.join(os.path.join(BASE_PREDICTIONS_DIR, projectname),file_name.split('.')[0])+".txt", "w") as text_file:
      text_file.write(jsondata)
  return "hii"





if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port='5000')

