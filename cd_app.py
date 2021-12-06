import sys
import numpy
import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import PIL
from tempfile import NamedTemporaryFile


img_size=200
st.title(''' Is it a cat or a dog ??''')
st.sidebar.header('Recognition')
my_image = st.sidebar.file_uploader(label='Upload your image for Recognition')


if my_image:
    st.image(my_image)

    img=PIL.Image.open(my_image)
    img_resized=img.resize((img_size,img_size))
    img_array=np.array(img_resized)[:,:,:3] #to eliminate the fourth  channel (transparency)
    X=img_array.reshape(1,img_size,img_size,3)

    #my_model =tf.keras.models.load_model('saved_model_100x100.h5')
    #my_model =tf.keras.models.load_model('MobileNet_alllayers_224x224')
    #my_model =tf.keras.models.load_model('model1_vgg16_100x100')
    my_model =tf.keras.models.load_model('modelVF1_vgg16_200x200')


    prediction=my_model.predict(X)
    result=['Cat','dog'][np.argmax(prediction)]
    st.write('## Prediction :')
    st.write('#### The image is most likely for a :',result)
    st.write('## Probabilities :')
    prediction=my_model.predict(X)
    st.write(pd.DataFrame(prediction, columns=['Cat','Dog']))
