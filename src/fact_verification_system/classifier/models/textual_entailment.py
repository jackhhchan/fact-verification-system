import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

def create_bert_model(max_seq_length):
    """
    inputs = [input_ids(max_seq_length, ), ]
    """
    assert max_seq_length <= 512 and max_seq_length > 0, "BERT can only handle a max sequence length of 512."
    
    # download BERT layer
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)
    
    # Specified Inputs
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                            name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                        name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                            name="segment_ids")

    pool_embs, all_embs = bert_layer([input_word_ids, input_mask, segment_ids])
    d_0 = Dense(units=512, activation='relu')(pool_embs)
    d_1 = Dense(units=256, activation='relu')(d_0)
    d_2 = Dense(units=1, activation='sigmoid', name='target')(d_1)

    return Model(inputs=[input_word_ids, input_mask, segment_ids],
                outputs=d_2)

def create_albert_model(max_seq_length):
    if tf.executing_eagerly():
        tf.compat.v1.disable_eager_execution()
    assert max_seq_length <= 512 and max_seq_length > 0, "BERT can only handle a max sequence length of 512."
    
    # download BERT layer
    albert_module = hub.Module(
                    "https://tfhub.dev/google/albert_base/3",
                    trainable=False)

    input_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                            name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                        name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                            name="segment_ids")


    # Specified Inputs
    albert_inputs = dict(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids
    )
    
    #__call__ method
    albert_output = albert_module(albert_inputs, signature='tokens', as_dict=True)
    pool_embs = albert_output['pooled_output']
    sequence_embs = albert_output['sequence_output']
    d_0 = Dense(units=512, activation='relu')(pool_embs)
    d_1 = Dense(units=256, activation='relu')(d_0)
    d_2 = Dense(units=1, activation='sigmoid', name='target')(d_1)

    return Model(inputs=[input_ids, input_mask, segment_ids],
                outputs=d_2)

if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()

    input_ids = [[1, 2, 3, 4, 5]]
    input_mask = [[0, 0, 0, 0, 0]]
    segment_ids = [[0, 0, 0, 0, 0]]

    input_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32,
                                            name="input_ids")
    input_mask = tf.keras.layers.Input(shape=(128,), dtype=tf.int32,
                                        name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32,
                                            name="segment_ids")


    albert_module = hub.Module(
                    "https://tfhub.dev/google/albert_base/3",
                    trainable=False)

    # Specified Inputs
    albert_inputs = dict(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids
    )
    
    tf.compat.v1.enable_eager_execution()
    #__call__ method
    albert_output = albert_module(albert_inputs, signature='tokens', as_dict=True)
    print(albert_output)

    print(albert_output['pooled_output'])
    print(type(albert_output['pooled_output']))

