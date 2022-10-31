import pytest
from src import test


@pytest.mark.parametrize("text,expected", [('''I'm gonna be home soon and i don't want to 
                                               talk about this stuff anymore tonight, 
                                               k? I've cried enough today.''', True),
                                           ('I hate you', False)])
def test_data_path(text, expected):
    result = float(test.test(text).squeeze())
    result = round(result) >= 0.5
    assert result == expected
