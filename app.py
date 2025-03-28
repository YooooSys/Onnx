import streamlit as st
from detect import *
from PIL import Image

st.title("Onnx testing")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if st.button("Student's ID recognition"):

    if uploaded_file is not None:

        image_arr = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

        image, h, w = (Image_read(image_arr)[x] for x in range(3))

        image_tensor = Preproccess_image(image)

        image_out = Detect(image, session, image_tensor, h, w)
        image_out = Image.fromarray(image_out)
        
        st.image(image_out, caption='Result', use_container_width=True)

    else:
        st.write("Provide an image")