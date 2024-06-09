from langchain.llms import GooglePalm 
import time
def connect_llm_model(): 
    try:
        api_key = "AIzaSyDc9dLSWX0jerFioIn3OoYPaXpxi0qsNKY"
        llm_model = GooglePalm(google_api_key=api_key, temperature=0.9)
        return llm_model
    except NotImplementedError as e:
        raise e

def stream_data(answear):
    for word in answear.split(" "):
        yield word + " "
        time.sleep(0.04)

def generate_text(llm_model, question):
    poem = llm_model(question)
    return poem