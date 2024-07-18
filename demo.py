import os
import requests

from PIL import Image
import pandas as pd

import streamlit as st

from predictor import Predictor
from utils import read_default_configs

models ={
    'Resnet50 unfreeze backbone': {
        'name':'resnet50',
        'weight_url': 'https://drive.google.com/uc?id=1zM0UtKGbrCdvz_qTHwslAp4PgBIAJ13U&export=download&confirm=t'},
    'Vit b16 freeze backbone': {
        'name':'vit_b_16',
        'weight_url': 'https://drive.google.com/uc?id=1CG4CuZrD3lcHIzy2IEVmWWGr9xmry8ka&export=download&confirm=t'}
    }

@st.cache_data
def load_session():
    return requests.Session()

def predictor(cfg):
    return Predictor(cfg)

def file_selector(folder_path):
    filenames = os.listdir(folder_path)
    selected_filenames = st.multiselect('Or select some samples', filenames)
    return selected_filenames, [os.path.join(folder_path, file) for file in selected_filenames]

def main():
    st.set_page_config(
        page_title="Cat recognition",
        page_icon=":star:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title(":newspaper: Cat recognition")
    sess = load_session()
    uploaded_images = st.file_uploader("Choose images: ", accept_multiple_files=True)
    
    samples_image_names, samples_image_dirs = file_selector('samples/')
    
    model = st.selectbox(
        'Choose model',
        ('Resnet50 unfreeze backbone','Vit b16 freeze backbone'))
    
    configs = read_default_configs()
    configs['device'] = 'cpu'
    configs['model_name'] = models[model]['name']
    configs['weights'] = models[model]['weight_url']
    st.image(uploaded_images, caption = [image.name for image in uploaded_images])
    st.image(samples_image_dirs, caption=samples_image_names)
    if 'configs' not in st.session_state or st.session_state.configs != configs:
        st.session_state.configs = configs
        st.session_state.predictor = predictor(configs)
    button = st.button("Predict")

    st.markdown(
        "<hr />",
        unsafe_allow_html=True
    )
    if button:
        detector = st.session_state.predictor
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