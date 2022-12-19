import base64
import onnxruntime as ort
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
# from torch import softmax, argmax
# ------------- for testing ---------------
# test_img = './test_data/knife3.jpg'
# ------------- for testing ---------------


onnx_model = "./kitchenware_model.onnx"
labels_raw = pd.Series(['glass', 'cup', 'spoon', 'plate', 'knife', 'fork']) # These labels were obtained from EDA

le = LabelEncoder()
labels = le.fit_transform(labels_raw)


def model_loader():
    session = ort.InferenceSession(onnx_model)
    input_name = session.get_inputs()[0].name

    return input_name, session

def img_decoder(img_string):
    with open("imagefile.jpg", "wb") as f:
        f.write(base64.decodebytes(img_string))
    return "imagefile.jpg"


def preprocess(input_data):
    # convert the input data into the float32 input
    img_data = input_data.astype('float32')

    #normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]

    norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')
    return norm_img_data

def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def postprocess(result):
    return softmax(np.array(result))


def classify(img):

    input_name, session = model_loader()
    # img_decoded = img_decoder(img)

    img_proc = Image.open(img.stream).resize((224, 224))
    img_data = np.array(img_proc, dtype=np.float32).transpose(2, 0, 1)
    input_data = preprocess(img_data)   

    result = session.run(None, {input_name: input_data})

    soft_max = postprocess(result)
    label = np.argmax(soft_max)
    transformed_label = le.inverse_transform(np.array(label, ndmin=1))
    output = transformed_label

    return output[0]

    
