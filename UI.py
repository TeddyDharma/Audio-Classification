import streamlit as st 

st.header("welcomce to audio classifier", devider = "bkue")
st.subheader("creted by data science students at Bisa ai")

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None: 
    # next : call the extration functions