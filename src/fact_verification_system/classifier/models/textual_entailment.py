import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

def create_model(max_seq_length):
    """
    inputs = [input_ids(max_seq_length, ), ]
    """
    assert max_seq_length <= 512 and max_seq_length > 0, "BERT can only handle a max sequence length of 512."
    
    # download BERT layer
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1")
    
    # Specified Inputs
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                            name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                        name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                            name="segment_ids")

    pool_embs, all_embs = bert_layer([input_word_ids, input_mask, segment_ids])
    d_0 = Dense(units=16, activation='relu')(pool_embs)
    d_1 = Dense(units=1, activation='sigmoid', name='target')(d_0)

    return Model(inputs=[input_word_ids, input_mask, segment_ids],
                outputs=d_1)