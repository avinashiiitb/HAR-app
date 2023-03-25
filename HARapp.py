import streamlit as st
import PIL
from PIL import Image
import io
import tensorflow as tf
import numpy as np

def load_image():
    upload_file = st.file_uploader(label ='pick a test image')
    if upload_file is not None:
        image_data = upload_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None
def load_model():
    model = tf.keras.models.load_model('model_1008_.h5')

    return model

categories = ['00','01','02','03','04','05','06','07','08','09','10']


def predict(model, categories, image):
    image = np.array(image)
    image = np.expand_dims(image,axis=0)
    img = tf.image.resize(image,(128,128))
    img = tf.cast(img,tf.float32)
    img = (img -70.55581) /86.36526
    
   



    prob = model.predict(img)
    

    
    #for i in range(prob.shape[0]):
     #   st.write(categories[i],prob[i])

 st.write(prob) 




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
