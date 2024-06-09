import streamlit as st 
from LLM import * 
from  model.model import * 
import json
st.header("Welcomce to audio classifier")
st.subheader("Creted by data science students at Bisa AI Academy")

with st.expander('About this app'):
  st.markdown('**What can this app do?**')
  st.info('This app can classify the audio file, based on the features extraction]')

  st.markdown('**How to use the app?**')
  st.warning('You just only drag and drop your audio file, and walaaa  the deep learnig model will predict the class')

  st.markdown('**Under the hood**')
  st.markdown('Data sets:')
  st.code('''- Audio Dataset''', language='markdown')
uploaded_file = st.file_uploader("Choose a file", type =  ".wav")

if uploaded_file is not None: 
    bytes_data = uploaded_file.getvalue()
    tabular_pred = predict_tabular(uploaded_file)
    st.success("classfication result based on data tabular : " +  str(json.loads(tabular_pred)['prediction'])) 
    llm  = connect_llm_model()
    ans = generate_text(llm, f'describe about {json.loads(tabular_pred)["prediction"]} in two paragraphs')
    st.write_stream(stream_data(ans))
   

