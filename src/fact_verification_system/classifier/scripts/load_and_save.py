"""
Build model,
Load model weights (trained by gcp-vm)
Save model in protobuf format.
https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/saved_model
"""

import os
import tensorflow as tf

from fact_verification_system.classifier.models import textual_entailment as te

def main():
    max_seq_length=64
    model = te.create_bert_model(max_seq_length)

    print("Loading weights...")
    weights_path = "../trained_models/tf-2-vm/model_weights.hdf5"
    model.load_weights(weights_path)

    model.summary()

    MODEL_DIR = "saved_model"
    VERSION = 1
    export_path = os.path.join(MODEL_DIR, str(VERSION))

    tf.keras.models.saved_model(
        model,
        export_path,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )

    print("Model saved to {}".format(export_path))


if __name__ == "__main__":
    main()