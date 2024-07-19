import tensorflow as tf
import streamlit as st
from PIL import Image
from patch_generator import smash_n_reconstruct
from filters import apply_all_filters
from featureExtractionLayer import featureExtractionLayer


# Preprocessing wrapper

def preprocess_input_img(img):
    rt,pt = smash_n_reconstruct(img)
    frt = tf.cast(tf.expand_dims(tf.expand_dims(apply_all_filters(rt),axis=-1),axis=0), dtype=tf.float64)
    fpt = tf.cast(tf.expand_dims(tf.expand_dims(apply_all_filters(pt), axis=-1),axis=0), dtype=tf.float64)

    return tuple([{
        'rich_texture':frt,
        'poor_texture':fpt
    }])

# Interface

def generate_ai_prob_score(input_image):
    preprocessed_img = preprocess_input_img(input_image)

    model = tf.keras.models.load_model('patch_craft_classifier.h5', custom_objects={'featureExtractionLayer': featureExtractionLayer})
    ai_score = model.predict(preprocessed_img)

    return ai_score

# GUI

st.title('AI-Generated Image Detection App')
st.write('Using PatchCraft Model (https://arxiv.org/pdf/2311.12397)')

uploaded_file = st.file_uploader("Insert an image")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image)

    with st.spinner('Wait for it...') :
        ai_score = generate_ai_prob_score(image)
        formatted_score = f'{round(ai_score[0][0] * 100)}%'

    if (ai_score != None) :
        st.divider()
        st.success('This is a success message!', icon="✅")
        st.write('Probability')
        st.code(formatted_score, language='bash')