import tensorflow as tf
import pandas as pd
from src.data_preprocess import (cleanText,tokenize_and_pad)

def test(text):
    model = tf.keras.models.load_model('./model/text/')
    clean_text = cleanText(text)
    token_pad,_ = tokenize_and_pad(pd.DataFrame([{"text": clean_text}]))
    predict = model.predict(token_pad)
    return predict