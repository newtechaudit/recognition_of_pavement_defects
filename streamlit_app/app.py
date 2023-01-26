import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import streamlit as st
import cv2

class_names = ['дефекты не найдены!', 'дефекты найдены!']


@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model('model.h5')


with st.spinner('Модель загружается..'):
    model = load_model()

html_temp = '''
    <div style='background-color:#1DB4AC ;padding:10px'>
    <h2 style='color:yellow;text-align:center;'>
    Распознавание дефектов дорожного полотна
    </h2>
    </div>
    '''
st.markdown(html_temp, unsafe_allow_html=True)

file = st.file_uploader(
    'Пожалуйста, загрузите изображение в формате *.jpg/*.jpeg/*.pdf/*.png',
    type=['jpg', 'jpeg', 'pdf', 'png']
)

st.set_option('deprecation.showfileUploaderEncoding', False)


def import_and_predict(image_data, model):
        size = (120,120)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_reshape = img[np.newaxis, ...]

        prediction = model.predict(img_reshape)
        return prediction


if file is None:
    st.text(
        'Пожалуйста, загрузите изображение в формате *.jpg/*.jpeg/*.pdf/*.png'
    )
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    if st.button('Распознавание'):
        predictions = import_and_predict(image, model)
        score=np.array(predictions[0])
        st.write(score)
        st.title('Результат: {}'.format(class_names[np.argmax(score)]))
