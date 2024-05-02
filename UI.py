import streamlit as st 
from main import * 
st.header("Welcomce to audio classifier")
st.subheader("Creted by data science students at Bisa ai")

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None: 
    # next : call the extration functions
    features_tabular = feature_extraction_tabular(uploaded_file)
    display_tabular_data(features_tabular)
    tabular = predict_tabular(features_tabular)
    st.success("classfication result based on data tabular : " +  str(tabular), icons = "🤖") 
    st.success("classfication result based melspectogram: " + str(), icons = "🤖") 
