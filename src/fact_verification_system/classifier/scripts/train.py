import os
import json
from enum import Enum
import multiprocessing
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD

from fact_verification_system.classifier.models import textual_entailment as te

## NOTE: Manually tune parameters in Hyperparams
class Hyperparams(Enum):
    BATCH_SIZE = 8
    EPOCHS = 80
#     OPTIMIZER = Adam(learning_rate=0.001)        # default 0.01
    OPTIMIZER = SGD(learning_rate=0.001)          # default 0.01
    LOSS = 'binary_crossentropy'
    METRICS = ['accuracy']
    BUFFER_SIZE = 1024
    
class CallbackParams(Enum):
    # earlystopping 
    MONITOR = 'val_loss'            # or 'loss' for training accuracy
    PATIENCE = 2
    # tensorboard
    HISTOGRAM_FREQ = 1              # default: 0  -- freq (in epochs) at which to compute activation and weight histograms
    UPDATE_FREQ = 'epoch'           # default: epoch   -- either batch or epoch or integer (100 = per 100 batches)

class Model(Enum):
    MAX_SEQ_LENGTH = 64        # NOTE: CHANGE THIS when dataset changes
                                #       affects BERT Layer and dataset parsing.

def main():    
    # Parallel Extraction
    # suffix = "train_64_slightly_more_supports.tfrecord"      #NOTE: CHANGE THIS
#     suffix_train = "train_64_balanced.tfrecord"
    suffix_train = "train_64_balanced_10000_samples.tfrecord"
    # suffix = "train_raw_string.tfrecord"
    file_pattern = "../dataset/tfrecords/" + suffix_train

    ds = _extract(file_pattern)

    # suffix = "devset" + suffix[5:]      #NOTE: CHANGE THIS
#     suffix_dev = "devset_64_balanced.tfrecord"
    suffix_dev = "devset_64_balanced_2000_samples.tfrecord"
    # suffix = "devset_raw_string.tfrecord"
    file_pattern = "../dataset/tfrecords/" + suffix_dev

    ds_val = _extract(file_pattern)

    # Parallel Parsing & Transformation
    print("Parsing TFRecords into dataset...")
    num_cpus = multiprocessing.cpu_count()

    ds = ds.map(_parse_and_transform, num_parallel_calls=num_cpus)          # NOTE: CHANGE THIS
    ds_val = ds_val.map(_parse_and_transform, num_parallel_calls=num_cpus)  # NOTE: CHANGE THIS

    # Cached
    ds = ds.cache()
    ds_val = ds_val.cache()
    print("Dataset cached to memory.")
    # Parallel Loading
    ds = ds.shuffle(Hyperparams.BUFFER_SIZE.value).batch(Hyperparams.BATCH_SIZE.value)
    ds_val = ds_val.shuffle(Hyperparams.BUFFER_SIZE.value).batch(Hyperparams.BATCH_SIZE.value)

    print("Dataset shuffled and batched into {}s".format(Hyperparams.BATCH_SIZE.value))
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)
    print("Dataset prefetched into buffer.")
    
    # Model
    print("[IMPORTANT] {} max length used to build model.".format(Model.MAX_SEQ_LENGTH.value))
    model = te.create_bert_model(max_seq_length=Model.MAX_SEQ_LENGTH.value)
    # model = te.create_bilstm_model()                                        # NOTE: CHANGE THIS

    model.compile(optimizer=Hyperparams.OPTIMIZER.value,
                loss=Hyperparams.LOSS.value,
                metrics=Hyperparams.METRICS.value)
    model.summary()

    
    # Training
    history_dir = "models_history/"
    if not os.path.isdir(history_dir):
        os.makedirs(history_dir)

    timestamp = datetime.now().strftime("%d.%m.%Y-%H.%M.%S")

    log_dir = history_dir + timestamp
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    notes = input("Additional training notes:")
    if notes:
        with open(log_dir + '/additional-notes.txt', 'w') as fh:
            fh.write(notes)

    # callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, 
                                                        histogram_freq=CallbackParams.HISTOGRAM_FREQ.value,
                                                        update_freq=CallbackParams.UPDATE_FREQ.value)
    earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor=CallbackParams.MONITOR.value, 
                                                              patience=CallbackParams.PATIENCE.value)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=log_dir+'/model_weights.hdf5',
                                                                save_best_only=True,
                                                                save_weights_only=True)

    print("Writing to {}...".format(log_dir))
    _ = model.fit(x=ds,
            epochs=Hyperparams.EPOCHS.value,
            callbacks=[tensorboard_callback, earlystopping_callback, model_checkpoint_callback],
            validation_data=ds_val,
            verbose=1)
            
    config_dir = log_dir + "/config/"
    if not os.path.isdir(config_dir):
        os.makedirs(config_dir)
    json_config = model.to_json()
    with open(config_dir + 'bert_model_config.json', 'w') as jf:
        jf.write(json_config)

    with open(log_dir + '/model_summary.txt','w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    with open(log_dir + '/other_info.txt', 'w') as fh:
        fh.write("Datasets used:\n{}\n{}\n".format(suffix_train, suffix_dev))
        fh.write("Optimizer: {}".format(str(Hyperparams.OPTIMIZER.value)))


    print("Model history saved to {}.".format(log_dir))


def _extract(file_pattern:str) -> tf.data.Dataset:
    print("Reading from file pattern: {}".format(file_pattern))
    if not input("Confirm? (y/n): ") == 'y':
            exit("Terminated.")
    files = tf.data.Dataset.list_files(file_pattern)
    
    ds = files.interleave(lambda x:
                    tf.data.TFRecordDataset(x),
                    cycle_length=4,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    print("Extracted all TFRecords with pattern {}".format(file_pattern))
    return ds


def _parse_and_transform(serialized)-> dict:
    feature_description = {
        'input_word_ids': tf.io.FixedLenFeature([Model.MAX_SEQ_LENGTH.value], tf.int64),
        'input_mask': tf.io.FixedLenFeature([Model.MAX_SEQ_LENGTH.value], tf.int64),
        'segment_ids': tf.io.FixedLenFeature([Model.MAX_SEQ_LENGTH.value], tf.int64),
        'target': tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(serialized, feature_description)
    # transform
    target = example.pop('target')
    target = tf.reshape(target, ())
    # target = tf.cast(target, tf.float32)
    
    embeddings_dict = example
    for k, v in embeddings_dict.items():
        embeddings_dict[k] = tf.cast(v, tf.float32)

    target_dict = {'target': target}

    return (embeddings_dict, target_dict)


def _parse_and_transform_str(serialized)-> dict:
    feature_description = {
        'input_0': tf.io.VarLenFeature(dtype=tf.string),
        'input_1': tf.io.VarLenFeature(dtype=tf.string),
        'target': tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(serialized, feature_description)
    # transform
    target = example.pop('target')
    target = tf.reshape(target, ())
    
    string_dict = example
    print(tf.train.Example().ParseFromString(serialized.numpy()))
    x = tf.io.decode_raw(string_dict['input_0'], output_type='half')
    # x = string_dict['input_0']
    print(x)
    target_dict = {'target': target}

    return (string_dict, target_dict)


if __name__ == "__main__":
    main()