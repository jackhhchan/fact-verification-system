"""
Build model,
Load model weights (trained by gcp-vm)
Save model in SavedFormat(protobuf) format.
https://www.tensorflow.org/tfx/tutorials/serving/rest_simple#make_a_request_to_your_model_in_tensorflow_serving

SavedFormat is used in TFX (tensorflow-serving).
"""

import os
import tensorflow as tf

from fact_verification_system.classifier.models import textual_entailment as te

def main():
    max_seq_length=64
    model = te.create_bert_model(max_seq_length)

    print("Loading weights...")
    WEIGHTS_DIR = "../trained_models/desktop/10000samples"
    WEIGHTS_FNAME = "model_weights.hdf5"
    model.load_weights(os.path.join(WEIGHTS_DIR, WEIGHTS_FNAME))

    model.summary()

    MODEL_DIR = "ModelSavedFormat"
    VERSION = 1
    export_path = os.path.join(WEIGHTS_DIR, MODEL_DIR, str(VERSION))

    tf.keras.models.save_model(
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