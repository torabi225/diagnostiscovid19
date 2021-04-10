# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 00:40:50 2020

@author: ASUS
"""

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

def import_and_predict(image_data, model):
    
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = image.convert('RGB')
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)

        img_reshape = image[np.newaxis,...]

        prediction = model.predict(img_reshape)
        
        return prediction

model = tf.keras.models.load_model('covid19_xray.h5')

st.write("""
       تشخیص بیماری کوید 19 از روی تصاویر رادیولوژی
         """
         )

st.write("")

file = st.file_uploader("لطفا یک تصویر رادیولوژی اپلود کنید", type=["jpg", "png","jpeg"])
#
if file is None:
    st.text("شما هنوز عکسی اپلود نکرده اید")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("covid!")
    elif np.argmax(prediction) == 1:
        st.write("normal!")
    
    st.text("Probability (0: covid, 1: normal)")
    st.write(prediction)