import pytest
from src import data_preprocess
import pandas as pd
import numpy as np


@pytest.mark.parametrize("path,expected", [("./db/spam.csv", True), ('', False)])
def test_data_path(path, expected):
    result = data_preprocess.get_dataset(path)
    result = len(result) > 0
    assert result == expected


@pytest.mark.parametrize("s,expected", [("Hello@@make999", "hello"), ("---99", "")])
def test_clean_text(s: str, expected):
    result = data_preprocess.cleanText(s)
    assert result == expected


@pytest.mark.parametrize("df,expected", [(pd.DataFrame([{"text": '''Go until jurong point, crazy.. Available 
                                                                    only in bugis n great world la e buffet...
                                                                    Cine there got amore wat...'''}]), True),
                                        (pd.DataFrame([{"text": ""}]), False)])
def test_token_pad(df, expected):
    data,_ = data_preprocess.tokenize_and_pad(df)
    result = sum(data[0]) > 0 
    assert result == expected
