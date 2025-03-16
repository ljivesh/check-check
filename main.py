import streamlit as st
from PIL import Image

# Title
st.title("Display Image from Google Drive")

# Input for image path
image_path = st.text_input("https://drive.google.com/file/d/1pgr2ZPOQ8WXyVypisuEavnTDVvmY04bs/view?usp=drive_link")

if image_path:
    try:
        # Open and display image
        image = Image.open(image_path)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    except Exception as e:
        st.error(f"Error loading image: {e}")
