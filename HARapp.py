import streamlit as st
import PIL
from PIL import Image
import io
import tensorflow as tf
import numpy as np
import  cv2
import pandas as pd
import plotly.express as px

def load_image():
    upload_file = st.file_uploader(label ='pick a test image')
    if upload_file is not None:
        image_data = upload_file.getvalue()
        st.image(image_data)
        
        file_bytes=np.asarray(bytearray(upload_file.read()), dtype=np.uint8)
        opencv_image=cv2.imdecode(file_bytes,1)
        return opencv_image
    else:
        return None
def load_model():
    model = tf.keras.models.load_model('model_1008_.h5')

    return model

categories = ['00','01','02','03','04','05','06','07','08','09','10']


def predict(model, categories, image):
    #image = np.array(image ,dtype = np.float32)
    image = tf.cast(image,tf.float32)
    image = np.expand_dims(image,axis=0)
    img = tf.image.resize(image,(128,128))
    #img = tf.cast(img,tf.float32)
    img = (img -70.55581) /86.36526
    
   



    prob = model.predict(img)
    
   
    prob = np.array(prob)


    st.write(prob) 
    
    chart_data = pd.DataFrame(
    prob,
    columns=categories)
    
    fig = px.bar(
    chart_data,
    x="categories",
    y="probabilities",
    
)

  
    st.bar_chart(chart_data)
    st.plotly_chart(fig)



def main():
    st.title('Custom model demo')
    model = load_model()
    image = load_image()
    result = st.button('Run on image')
    if result:
        st.write('Calculating results...')
        predict(model, categories, image)


if __name__ =='__main__':
    main()
