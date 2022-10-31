import re
import pandas as pd
import tensorflow as tf


def get_dataset(path):
    try:
        df = pd.read_csv(path, encoding='latin-1')
        df = df.iloc[:, :2]
        df.columns = ['label', 'text']
        return df
    except Exception as e:
        print("Not valid Path")
        return pd.DataFrame()


def get_data(df):
    df = df.replace('spam', 1)
    df = df.replace('ham', 0)
    return df


def cleanText(text):
    whitespace = re.compile(r"\s+")
    web_address = re.compile(r"(?i)http(s):\/\/[a-z0-9.~_\-\/]+")
    user = re.compile(r"(?i)@[a-z0-9_]+")
    text = text.replace('.', '')
    text = whitespace.sub(' ', text)
    text = web_address.sub('', text)
    text = user.sub('', text)
    text = re.sub(r"\[[^()]*\]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r"(?:@\S*|#\S*|http(?=.*://)\S*)", "", text)
    return text.lower()


def tokenize_and_pad(df):
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.oov_token = '<oovToken>'
    tokenizer.fit_on_texts(df.text)
    data = tf.keras.preprocessing.sequence.pad_sequences(
        tokenizer.texts_to_sequences(df.text.to_numpy()), padding='pre', maxlen=171)
    vocab = tokenizer.word_index
    vocabCount = len(vocab)+1
    return data,vocabCount

def split_data(X,df,split_size):
    y = df.label.to_numpy()
    dim = X.shape[1]
    xTest = X[split_size:]
    yTest = y[split_size:]
    xTrain = X[:split_size]
    yTrain = y[:split_size]
    return xTrain,yTrain,xTest,yTest,dim
    


