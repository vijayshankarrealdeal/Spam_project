import pytest
from src import model
import tensorflow as tf

@pytest.mark.parametrize("vocabCount,dim,expected", [(2,3,tf.keras.models.Sequential)])
def test_model(vocabCount,dim,expected):
    network = model.network_model(vocabCount,dim)
    if network == expected:
        assert network == expected

def test_value_error():
    with pytest.raises(ValueError):
        model.network_model(-1,-1)
