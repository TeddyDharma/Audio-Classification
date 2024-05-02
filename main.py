# creating melsppectogram function 


import tensorflow as tf
import os
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt

# predict using melspectogram image


def get_melspectogram(audio): 
    audio, sr  = librosa.load(audio)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    fig, ax = plt.subplots(facecolor='none')
    S_dB = librosa.power_to_db(audio, ref=np.max)
    img = librosa.display.specshow(mel, sr=sr)
    return img


def predict_image(audio): 
    classes = [x for  x in os.listdir("./data/genres_original/")]
    model = tf.keras.models.load_model("./image_classfication.h5")
    mel = get_melspectogram(audio)
    raw = tf.io.read_file(image)
    image = tf.image.decode_png(raw, channels=3)
    pred = classes[np.argmax(model.predict([image]))]
    return pred


# getting average and sum from feature extraction
def get_value(array : np.array, action):
    array_flat = array.flatten()
    if action == "avg": 
        return np.mean(array_flat)
    elif action  == "sum": 
        return np.sum(array_flat)
    else:
        raise get_value("please check again your action")

def feature_extraction_tabular(input_audio): 
    array_avg_spectral_bandwidth = []
    array_avg_zero_crossing_rate = []
    arrray_avg_rms = []
    array_avg_spectiaL_centroid = []
    array_avg_spectiaL_bandwith = []
    array_avg_spectral_rolloff = []
    array_avg_spectral_flattness = []

    # sum array
    array_sum_spectral_bandwidth = []
    array_sum_zero_crossing_rate = []
    arrray_sum_rms = []
    array_sum_spectiaL_centroid = []
    array_sum_spectiaL_bandwith = []
    array_sum_spectral_rolloff = []
    array_sum_spectral_flattness = []
    audio, sr  = librosa.load(input_audio)
    # chroma features
    array_avg_spectral_bandwidth.append(get_value(librosa.feature.spectral_bandwidth(y=audio, sr=sr), action="avg"))
    # zero crossing rate
    array_avg_zero_crossing_rate.append(get_value(librosa.feature.zero_crossing_rate(y  = audio), action="avg"))
    # rms
    arrray_avg_rms.append(get_value(librosa.feature.rms(y = audio), action="avg")) 
    # spectral centroid
    array_avg_spectiaL_centroid.append(get_value(librosa.feature.spectral_centroid(y = audio), action="avg"))
    # spectral bandwith
    array_avg_spectiaL_bandwith.append(get_value(librosa.feature.spectral_bandwidth(y = audio, sr = sr), action="avg"))
    # spectral roll off
    array_avg_spectral_rolloff.append(get_value(librosa.feature.spectral_rolloff(y = audio, sr = sr), action="avg"))
    # speectral flattness
    array_avg_spectral_flattness.append(get_value(librosa.feature.spectral_flatness(y = audio), action="avg"))
    # for labels
    # adding another method



    # sum
    array_sum_spectral_bandwidth.append(get_value(librosa.feature.spectral_bandwidth(y=audio, sr=sr), action="sum"))
    # zero crossing rate
    array_sum_zero_crossing_rate.append(get_value(librosa.feature.zero_crossing_rate(y  = audio), action="sum"))
    # rms
    arrray_sum_rms.append(get_value(librosa.feature.rms(y = audio), action="sum")) 
    # spectral centroid
    array_sum_spectiaL_centroid.append(get_value(librosa.feature.spectral_centroid(y = audio), action="sum"))
    # spectral bandwith
    array_sum_spectiaL_bandwith.append(get_value(librosa.feature.spectral_bandwidth(y = audio, sr = sr), action="sum"))
    # spectral roll off
    array_sum_spectral_rolloff.append(get_value(librosa.feature.spectral_rolloff(y = audio, sr = sr), action="sum"))
    # speectral flattness
    array_sum_spectral_flattness.append(get_value(librosa.feature.spectral_flatness(y = audio), action="sum"))

    dict_features_extraction = {
        "avg spactial bandwith" : array_avg_spectral_bandwidth, 
        "avg zero crossing rate" : array_avg_zero_crossing_rate,
        "avg rms" : arrray_avg_rms, 
        "avg spactial centroid" : array_avg_spectiaL_centroid,
        "avg spectral flattness" : array_avg_spectral_flattness, 
        "avg spectrall rolloff" : array_avg_spectral_rolloff, 
        # sum
        "sum spactial bandwith" : array_sum_spectral_bandwidth, 
        "sum zero crossing rate" : array_sum_zero_crossing_rate,
        "sum rms" : arrray_sum_rms, 
        "sum spactial centroid" : array_sum_spectiaL_centroid,
        "sum spectral flattness" : array_sum_spectral_flattness, 
        "sum spectrall rolloff" : array_sum_spectral_rolloff, 
    }
    return pd.DataFrame(dict_features_extraction)


def display_tabular_data(data_audio):
    print(feature_extraction_tabular(data_audio))

def predict_tabular(data_audio): 
    data_extraction = feature_extraction_tabular(data_audio)
    pred_data = np.array([x for x in data_extraction[0 : 1 : len(data_extraction.columns)]])
    model = tf.keras.models.load_model("./tabular_classfication.h5")
    tabular_pred = model.predict([pred_data])
    return tabular_pred