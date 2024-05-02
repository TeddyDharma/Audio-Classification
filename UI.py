import streamlit as st 
from main import * 
st.header("Welcomce to audio classifier")
st.subheader("Creted by data science students at Bisa ai")

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None: 
    # next : call the extration functions
    features_tabular = feature_extraction_tabular(uploaded_file)
    # st.checkbox("Use container width", value=False, key="use_container_width") 
    # tabular_norm = predict_tabular(features_tabular)
    st.dataframe(features_tabular, height = 100)
    # tabular_pred = predict_tabular(uploaded_file)
    # st.success("classfication result based on data tabular : " +  str(tabular_pred), icons = "🤖") 
    # st.success("classfication result based melspectogram: " + str(), icons = "🤖") 

