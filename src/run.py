from src import data_preprocess
from src import train


def run_train(train:bool):
    df = data_preprocess.get_dataset("./db/spam.csv")
    df = data_preprocess.get_data(df)
    df.text = df.text.apply(lambda x : data_preprocess.cleanText(x))
    data,vocab_size = data_preprocess.tokenize_and_pad(df)
    x_train,y_train,x_test,y_test,dim = data_preprocess.split_data(data,df,5000)
    history = train.train(x_train,y_train,x_test,y_test,vocab_size,dim)
    return history