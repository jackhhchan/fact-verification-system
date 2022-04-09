# Textual Entailment
"""
Textual Entailment classifier


"""

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


class TextualEntailment(Model):
    def __init__(self):
        super(TextualEntailment, self).__init__()
        self.BERT = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1")
        self.d_0 = Dense(units=16, activation='relu')
        self.d_1 = Dense(units=1, activation='sigmoid', name='target')

    def call(self, inputs):
        # TODO: test inputs


        pool_embs, _ = self.Bert([inputs])      # _ = list of embs of each word from BERT model
        x = self.d_0(pool_embs)
        x = self.d_1(x)
        return x

    def checkInputs(self, inputs):
        # checks BERT inputs
        return True




if __name__ == "__main__":
    te = TextualEntailment()

    # ds = tf.Dataset.data()
    # NOTE: create dataset here to check model

    te.fit(x=None)