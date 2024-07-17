import os
import requests

from PIL import Image
import pandas as pd

import streamlit as st

from predictor import Predictor
from utils import cfg, read_default_configs

models ={
    'Vit-b32 freeze backbone': {
        'name':'vit_b_32',
        'weight_url': 'https://drive.google.com/uc?id=1AQAaqWHERvdg9lsQqWG82V6rjnVr6Qq0&export=download&confirm=t'}
    }

@st.cache_data
def load_session():
    return requests.Session()

@st.cache_resource
def predictor(_cfg):
    return Predictor(_cfg)

# cinnamon_dataset_url = 'https://drive.google.com/drive/folders/1Qa2YA6w6V5MaNV-qxqhsHHoYFRK5JB39'
# vnondb_word_url = 'https://tc11.cvc.uab.es/datasets/HANDS-VNOnDB2018_1/'

def file_selector(folder_path):
    filenames = os.listdir(folder_path)
    selected_filenames = st.multiselect('Or select some samples', filenames)
    return selected_filenames, [os.path.join(folder_path, file) for file in selected_filenames]

def main():
    st.set_page_config(
        page_title="Vietnamese handwriting recognition",
        page_icon=":star:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title(":newspaper: Vietnamese handwriting recognition")
    sess = load_session()
    # st.write(f"You can download [Cinnamon handwritting dataset]({cinnamon_dataset_url}) or [VNOnDB-Word]({vnondb_word_url})")
    uploaded_images = st.file_uploader("Choose images (images must come from same dataset): ", accept_multiple_files=True)
    
    samples_image_names, samples_image_dirs = file_selector('samples/')
    
    model = st.selectbox(
        'Choose model',
        ('Vit-b32 freeze backbone',))
    
    default_configs = read_default_configs()
    default_configs['device'] = 'cpu'
    configs = cfg(default_configs)
    configs.model_name = models[model]['name']
    configs.weights = models[model]['weight_url']
    
    st.image(uploaded_images, caption = [image.name for image in uploaded_images])
    st.image(samples_image_dirs, caption=samples_image_names)

    button = st.button("Predict")

    st.markdown(
        "<hr />",
        unsafe_allow_html=True
    )
    if button:
        detector = predictor(configs)
        with st.spinner("Predicting..."):
            if len(uploaded_images) == 0 and len(samples_image_dirs) == 0:
                st.markdown('Please upload (an) images')
            else:
                name_list = list()
                text_list = list()
                for image in uploaded_images:
                    name_list.append(image.name)
                    text_list.append(detector.predict(Image.open(image)))
                for name,dir in zip(samples_image_names,samples_image_dirs):
                    name_list.append(name)
                    text_list.append(detector.predict(Image.open(dir)))
                st.table(pd.DataFrame({'Image name': name_list,
                                       'Text':text_list}))


if __name__ == "__main__":
    main()