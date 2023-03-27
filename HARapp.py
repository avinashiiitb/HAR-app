import streamlit as st
import PIL
from PIL import Image
import io
import os
import tensorflow as tf
import numpy as np
import  cv2
import pandas as pd
import plotly.express as px
import zipfile
from zipfile import *

def save_uploadedfile(uploadedfile):
     with open(os.path.join(".tempDir/",uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return st.success("Saved File:{} to tempDir".format(uploadedfile.name))





     
def load_image():
    upload_file = st.file_uploader(label ='Upload a test image')
   
    if upload_file is not None:
        image_data = upload_file.getvalue()
        save_uploadedfile(upload_file)
        st.image(image_data)   
        
        
        finished_main = st.button("Finish taking pictures")
    # Trying to add the zip file
        if finished_main:
          zipObj = ZipFile("sample.zip", "w")
          image_data = upload_file.getvalue()
          image1 = Image.open(io.BytesIO(image_data)) 
          
          with open('.tempDir/04000000_1574696114_Raw_0.png', 'rb') as f:
               img_to_zip = f.read()
          img_open = Image.open(io.BytesIO(img_to_zip))
          st.write(img_open)
          zipObj.writestr(zinfo_or_arcname='.tempDir/04000000_1574696114_Raw_0.png', data=img_to_zip)
                #zipObj.write(img_open)
          zipObj.close()
          ZipfileDotZip = "sample.zip"
            
          with open(ZipfileDotZip, "rb") as f:
               bytes = f.read()
               b64 = base64.b64encode(bytes).decode()
               href = f"<a href=\"data:file/zip;base64,{b64}\" download='{ZipfileDotZip}.zip'>\
                    Click last model weights\</a>"
          st.markdown(href, unsafe_allow_html=True)
     
     
     
     
     
        file_bytes=np.asarray(bytearray(upload_file.read()), dtype=np.uint8)
        opencv_image=cv2.imdecode(file_bytes,1)
        return opencv_image
    else:
        return None
def load_model():
    model = tf.keras.models.load_model('model_1008_.h5')

    return model

categories = ['00','01','02','03','04','05','06','07','08','09','10']
import base64

@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: contain;
    background-repeat: no-repeat;
    background-attachment: scroll; # doesn't work
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('background-1.png')

def predict(model, categories, image):
    image = np.array(image)
    image = tf.cast(image,tf.float32)
    image = np.expand_dims(image,axis=0)
    img = tf.image.resize(image,(128,128))
    
    img = (img -70.55581) /86.36526
    
   



    prob = model.predict(img)
    
   
    prob = np.array(prob)
   
    st.write(prob)
    prob = np.reshape(prob,(prob.shape[1],))


    st.write('Bar Graph of Probability') 
    d= {'Probability of each class' : prob , 'Classes':['00','01','02','03','04','05','06','07','08','09','10']}
    chart_data = pd.DataFrame(d)
    fig = px.bar(
    d,
    x="Classes",
    y="Probability of each class",
    color="Classes",
    text="Probability of each class",)
    
    st.plotly_chart(fig)
    
    st.write('Pie Chart of Probability')
    fig = px.pie(d, values='Probability of each class', names='Classes')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig)
    



def main():
    st.title('Human Activity Recognition Using Spectrograms')
    model = load_model()
    image = load_image()
    result = st.button('Run inference on image')
    if result:
        st.write('Probability Distribution')
        predict(model, categories, image)


if __name__ =='__main__':
    main()
