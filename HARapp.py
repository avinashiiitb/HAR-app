import streamlit as st
from deta import Deta
import matplotlib.pyplot as plt
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
import json

     
def load_image():
    upload_file = st.file_uploader(label ='Upload a test image')
   
    if upload_file is not None:
        image_data = upload_file.getvalue()
       
        st.image(image_data) 
        
        data = {}
        with open(os.path.join(".tempDir/",upload_file.name), 'rb') as file:
          img = file.read()
        data['img'] = base64.encodebytes(img).decode('utf-8')
        json.dumps(data)
        with st.form("form"):
          name = st.text_input("Enter Class number for activity recognition")
          submitted = st.form_submit_button("For storing in deta-db,press here ")
        deta = Deta(st.secrets["data_key"])
        db = deta.Base("HAR-db")
        if submitted:
          db.put({"name": name,'image':data})
        st.write("Here's everything stored in the database:")
        db_content = db.fetch().items
        st.write(db_content)
           
        
        finished_main = st.button("Press to prepare for image file download")
    # Trying to add the zip file
        if finished_main:
          zipObj = ZipFile("sample.zip", "w")
          image_data = upload_file.getvalue()
          image1 = Image.open(io.BytesIO(image_data)) 
          
          with open(os.path.join(".tempDir/",upload_file.name), 'rb') as f:
               img_to_zip = f.read()
          img_open = Image.open(io.BytesIO(img_to_zip))
          st.write(img_open)
          zipObj.writestr(zinfo_or_arcname='os.path.join(".tempDir/",upload_file.name)', data=img_to_zip)
                
          zipObj.close()
          ZipfileDotZip = "sample.zip"
            
          with open(ZipfileDotZip, "rb") as f:
               bytes = f.read()
               b64 = base64.b64encode(bytes).decode()
               href = f"<a href=\"data:file/zip;base64,{b64}\" download='{ZipfileDotZip}.zip'>\
                    Click for image download"
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
    text="Probability of each class",color_discrete_sequence=[
        "#0068c9",
        "#83c9ff",
        "#ff2b2b",
        "#ffabab",
        "#29b09d",
        "#7defa1",
        "#ff8700",
        "#ffd16a",
        "#6d3fc0",
        "#d5dae5"],)
    fn = 'bargraph.png'
    
    st.plotly_chart(fig)
    
    buffer = io.StringIO()
    fig.write_html(buffer, include_plotlyjs='cdn')
    html_bytes = buffer.getvalue().encode()
    
    st.download_button(
            label='Download Bar Graph HTML',
            data=html_bytes,
            file_name='bargraph.html',
            mime='text/html'
        )
    
    st.write('Pie Chart of Probability')
    fig = px.pie(d, values='Probability of each class', names='Classes')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig)
     
    buffer = io.StringIO()
    fig.write_html(buffer, include_plotlyjs='cdn')
    html_bytes = buffer.getvalue().encode()
    
    st.download_button(
            label='Download Pie chart HTML',
            data=html_bytes,
            file_name='piechart.html',
            mime='text/html'
        )
    



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
