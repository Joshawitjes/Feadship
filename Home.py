# To run type in terminal:
# cd Tool_App 
# python -m streamlit run script_hello.py
import streamlit as st
from streamlit import __main__
import os

st.title("Welcome to the Ship Analytics App")
st.write(os.getcwd())

image = "Feadship_yacht.jpg"
st.markdown("""
Use the sidebar to navigate:
- Upload your dataset
- Run OLS regression
- Explore and evaluate your model
""")

st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto", use_container_width=False)
