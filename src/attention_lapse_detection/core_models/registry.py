from attention_lapse_detection.core_models.gru import GRU
from attention_lapse_detection.core_models.gru_uni_attention import GRUUniAttention
from attention_lapse_detection.core_models.lstm import LSTM
from attention_lapse_detection.core_models.lstm_uni_attention import LSTMUniAttention


CLASSIFIERS = {
    "gru": GRU,
    "lstm": LSTM,
    "gru_uni_attention": GRUUniAttention,
    "lstm_uni_attention": LSTMUniAttention,
}

CLASSIFIERS_BY_CLASS_NAME = {cls.__name__: cls for cls in CLASSIFIERS.values()}
