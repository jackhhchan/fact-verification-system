#!/usr/bin/env python
# coding: utf-8

# ## Why does changing to SGD from Adam with a lower learning rate allow it to train?

# ### Tensorboard helper functions

# In[1]:


import tensorflow as tf
from datetime import datetime

def tensorboard_scalar(writer:tf.summary.SummaryWriter, 
                       name:str, data:float, step:int):
    with writer.as_default():
        tf.summary.scalar(name, data, step)
        
def tensorboard_histogram(writer:tf.summary.SummaryWriter, 
                       name:str, data:tf.Tensor, step:int):
    with writer.as_default():
        tf.summary.histogram(name, data, step)
        
def additional_notes(log_dir:str):
    notes = input("Additional training notes (Press enter to skip):")
    if notes:
        with open(log_dir + '/additional-notes.txt', 'w') as fh:
            fh.write(notes)

def model_summary_log(model: tf.keras.Model):
    with open(log_dir + '/model_summary.txt','w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

def datasets_log(train_fname:str, dev_fname:str):
    with open(log_dir + '/datasets.txt', 'w') as fh:
        fh.write("Datasets used:\n{}\n{}".format(suffix_train, suffix_dev))


# ### Metrics to log helper functions

# **Weights**

# In[2]:


### Scalars to log ###
def get_avg_min_max_neuron(neuron_params):
    """ returns a tuple of constants containing the weights of a neuron. """
    assert len(neuron_params.shape) == 1, "must only be for a single neuron."
    
    # get average of neuron weights
    avg = tf.math.reduce_mean(neuron_params)
    min_ = tf.math.reduce_min(neuron_params)
    max_ = tf.math.reduce_max(neuron_params)
    
    return (avg, min_, max_)


def get_avg_min_max_layer(layer_params_matrix):
    """ returns a tuple of tensors containing the params (i.e. weights or grads) of a layer.
    
    avg -- avg[0] = average param value for each input element for neuron 0.
    max_ -- max_[0] = max param value of every input element for neuron 0.
    min_ -- min_[0] = min param value of every input element for neuron 0.
    """
    lpm = layer_params_matrix
    num_neurons = layer_params_matrix.shape[1]  #[0] are input elements (i.e. prev layer neurons)
    
    layer_info = {
        'avg': list(),
        'min': list(),
        'max': list()
    }
    
    # curate avg, min and max neuron weight tensors for the layer
    for i in range(num_neurons):        
        (n_avg, n_min, n_max) = get_avg_min_max_neuron(lpm[:, i])
        layer_info['avg'].append(n_avg)
        layer_info['min'].append(n_min)
        layer_info['max'].append(n_max)

    avg = tf.convert_to_tensor(layer_info['avg'])
    min_ = tf.convert_to_tensor(layer_info['min'])
    max_ = tf.convert_to_tensor(layer_info['max'])
    
    assert avg.shape[0] == num_neurons,            "Avg neuron param tensor should be equivalent to number of neurons in the layer."
    
    return (avg, min_, max_)


# In[14]:


def tensorboard_hist_avg_min_max_for_layer(tensors_tuple:tuple, name_suffix:str, step:int):
    assert len(tensors_tuple) == 3, "There should be average, max and min tensors."
    (avg, min_, max_) = tensors_tuple
    tensorboard_histogram(writer, '{}_avg'.format(name_suffix), avg, step)
    tensorboard_histogram(writer, '{}_min'.format(name_suffix), min_, step)
    tensorboard_histogram(writer, '{}_max'.format(name_suffix), max_, step)


# **Gradients**

# In[4]:


##


# ### Data Pipeline

# In[5]:


max_seq_length =  64
def _parse_and_transform(serialized)-> dict:
    feature_description = {
        'input_word_ids': tf.io.FixedLenFeature([max_seq_length], tf.int64),
        'input_mask': tf.io.FixedLenFeature([max_seq_length], tf.int64),
        'segment_ids': tf.io.FixedLenFeature([max_seq_length], tf.int64),
        'target': tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(serialized, feature_description)
    # transform
    target = example.pop('target')
    target = tf.reshape(target, ())
#     target = tf.cast(target, tf.float32)
    
    embeddings_dict = example
    for k, v in embeddings_dict.items():
        embeddings_dict[k] = tf.cast(v, tf.int32)

    target_dict = {'target': target}

    return (embeddings_dict, target_dict)


# In[6]:


import os
os.path.exists("../dataset/tfrecords/train_64_balanced_1000_samples.tfrecord")


# In[8]:


import tensorflow as tf
from fact_verification_system.classifier.scripts.train import _extract
import multiprocessing

suffix_train = "train_64_balanced_10000_samples.tfrecord"
file_pattern = "../dataset/tfrecords/" + suffix_train

ds_train = _extract(file_pattern)

num_cpus = multiprocessing.cpu_count()
ds_train = ds_train.map(_parse_and_transform, num_parallel_calls=num_cpus)
ds_train = ds_train.cache()


# ### Model

# In[9]:


from fact_verification_system.classifier.models.textual_entailment import create_bert_model
model = create_bert_model(max_seq_length=64)

layer_indices = {l.name: i for i, l in enumerate(model.layers)}  ## to be used layer

model.summary()


# ### Training

# In[10]:


## for debugging training
import tensorflow as tf
def zero_grads_percentage(layer_grads:tf.Tensor):
    assert len(layer_grads.shape) == 2, "There should be gradients for each input element weight for each neuron."
    
    (avg, min_, max_) = get_avg_min_max_layer(layer_grads)
    zero_mask = tf.math.equal(avg, 0)
    layer_zero_grads_percent = (len(list(filter(lambda x: x, zero_mask))))/layer_grads.shape[1]
    return layer_zero_grads_percent


# In[11]:


### Training Step Function ### 
import tensorflow as tf
from tensorflow.keras.losses import Loss
from typing import Dict, List, Tuple

@tf.function
def compute_grads(train_batch: Dict[str,tf.Tensor], target_batch: tf.Tensor, 
                 loss_fn: Loss, model: tf.keras.Model):
    with tf.GradientTape(persistent=False) as tape:
        # forward pass
        outputs = model(train_batch)
        # calculate loss
        loss = loss_fn(target_batch, outputs)
    
    # calculate gradients for each param
    grads = tape.gradient(loss, model.trainable_variables)
    return grads, loss


# In[ ]:


### Training ###
import code
from IPython.core.debugger import set_trace
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam, SGD
from tqdm import tqdm

assert tf.executing_eagerly(), "Tensorflow not in eager execution mode."

# MANUAL
DATASET_SIZE = 10000
BATCH_SIZE = 8
EPOCHS = 15

bce = BinaryCrossentropy()
optimizer = SGD(learning_rate=0.001)

# tf.random.set_seed(1)  # reproducibility -- weights initialization

# tensorboard init
timestamp = datetime.now().strftime("%d.%m.%Y-%H.%M.%S")
train_log_dir = './train_logs_debug/{}'.format(timestamp)
writer = tf.summary.create_file_writer(train_log_dir)


# log initial weights
dense_0_w = model.layers[-3].weights[0]
avg_min_max = get_avg_min_max_layer(dense_0_w)
tensorboard_hist_avg_min_max_for_layer(avg_min_max, 'd_0_weights', 0)

dense_1_w = model.layers[-2].weights[0]
avg_min_max = get_avg_min_max_layer(dense_0_w)
tensorboard_hist_avg_min_max_for_layer(avg_min_max, 'd_1_weights', 0)


for epoch in tqdm(range(EPOCHS), desc='epoch'):
    # - accumulators
    epoch_loss = 0.0
    
    # - debug accumulators
    d_0_zero_grads = 0.0
    d_1_zero_grads = 0.0
    
    for (i, (train_batch, target_dict)) in tqdm(enumerate(ds_train.shuffle(1024).batch(BATCH_SIZE)), desc='step'):

        (grads, loss) = compute_grads(train_batch, target_dict['target'], bce, model)
        
        # debug -- track percentage of neurons receiving zero gradients
        d_0_zero_grads += zero_grads_percentage(grads[-4])
        d_1_zero_grads += zero_grads_percentage(grads[-2])
        
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        epoch_loss += loss
        if (i+1) % 250 == 0:
            print("average loss: {}".format(epoch_loss/(i+1)))
            
    avg_epoch_loss = epoch_loss/(i+1)
    tensorboard_scalar(writer, name='epoch_loss', data=avg_epoch_loss, step=epoch)
    print("Epoch {}: epoch_loss = {}".format(epoch, avg_epoch_loss))
    print(epoch)

    # % of zero gradients
    tensorboard_scalar(writer, name='d_0_zero_grads', data=d_0_zero_grads/(i+1), step=epoch)
    tensorboard_scalar(writer, name='d_1_zero_grads', data=d_1_zero_grads/(i+1), step=epoch)
    print("average dense_0 zero grad percentage: {}".format(d_0_zero_grads/(i+1)))
    print("average dense_1 zero grad percentage: {}".format(d_1_zero_grads/(i+1)))

    dense_0_w = model.layers[-3].weights[0]
    avg_min_max = get_avg_min_max_layer(dense_0_w)
    tensorboard_hist_avg_min_max_for_layer(avg_min_max, 'd_0_weights', step=(epoch+1))

    dense_1_w = model.layers[-2].weights[0]
    avg_min_max = get_avg_min_max_layer(dense_1_w)
    tensorboard_hist_avg_min_max_for_layer(avg_min_max, 'd_1_weights', step=(epoch+1))