"""
Tensorboard plotting utilities.
"""
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple


class ModelParams(object):
    """ Enhances model giving it some utility functions """

    def __init__(self, model:nn.Module):
        self.model = model

        # self.layer_names = (l_name for l_name in self.model._modules.keys())
        # self.param_names = (p_name for (p_name, _) in self.named_params)
        # self.params_dict = {p_name : param for p_name, param in self.named_params}
        # self.grads_dict = {p_name : param.grad for p_name, param in self.named_params}

    @property
    def named_params(self):
        return self.model.named_parameters()
    
    @property
    def layer_names(self):
        return (l_name for l_name in self.model._modules.keys())
    
    @property
    def param_names(self):
        return (p_name for (p_name, _) in self.named_params)

    @property
    def params_dict(self):
        return {p_name : param for p_name, param in self.named_params}

    def get_weights(self, layer_name:str):
        return self.params_dict.get("{}.weight".format(layer_name))
        
    def get_layer(self, layer_name:str) -> nn.Module:
        """ Returns the layer inside the model """
        return self.model._modules.get(layer_name)

class TensorboardUtil(object):
    def __init__(self, 
                model:ModelParams,
                writer:SummaryWriter):
        self.writer = writer
        self.model_params = model


        self.watched = False
        self.watched_layers = None

    def set_watch(self, layer_names:list):
        """" Sets which layers to watch. Also verifies inputted layer names."""
        assert self.model_params is not None, "ModelParams needed. Please update with update_model_params()."
        for l_name in layer_names:
            assert l_name in self.model_params.layer_names,\
                    "{} does not exist in model. It can't be watched.".format(l_name)
        self.watched_layers = layer_names
        self.watched = True
    
    # def update_model_params(self, model_params: ModelParams):
    #     self.model_params = model_params



class TensorboardHist(TensorboardUtil):
    def __init__(self,
                model:ModelParams,
                writer:SummaryWriter):
        super(TensorboardHist, self).__init__(model, writer)
        
        self.outputs_hooked = False
        self.outputs_cache = {}
        
    def plot_weights_hist(self, interval):
        """ plot weights histogram for watched layers. """

        assert self.watched, "Please set which layers to watch first using set_watch()."
        params_dict = self.model_params.params_dict
        for l_name in self.watched_layers:
            p_name = "{}.weight".format(l_name)
            try:
                self.writer.add_histogram(tag=p_name,
                                    values=params_dict.get(p_name),
                                    global_step=interval)
            except Exception as e:
                print("Failed to write weights histogram for layer: {}".format(l_name))
                raise e
    
    def plot_bias_hist(self, interval):
        """ plot bias histogram for watched layers. """

        assert self.watched, "Please set which layers to watch first using set_watch()."
        params_dict = self.model_params.params_dict
        for l_name in self.watched_layers:
            p_name = "{}.bias".format(l_name)
            try:
                self.writer.add_histogram(tag=p_name,
                                    values=params_dict.get(p_name),
                                    global_step=interval)
            except Exception as e:
                print("Failed to write bias histogram for layer: {}".format(l_name))
                raise e
    
    def plot_grads_hist(self, interval):
        """ plot grads histogram for watched layers. """

        assert self.watched, "Please set which layers to watch first using set_watch()."
        params_dict = self.model_params.params_dict
        for l_name in self.watched_layers:
            p_name_w = "{}.weight".format(l_name)
            p_name_b = "{}.bias".format(l_name)
            try:
                self.writer.add_histogram(tag=p_name_w+".grad",
                                    values=params_dict.get(p_name_w).grad,
                                    global_step=interval)
                self.writer.add_histogram(tag=p_name_b+".grad",
                                    values=params_dict.get(p_name_b).grad,
                                    global_step=interval)
                                
            except Exception as e:
                print("Failed to write grads histogram for layer: {}".format(l_name))
                print("Please check if gradients exist.")
                raise e

    # activations -- using hooks
    def register_outputs_hooks(self):
        """ registers forward hooks on activation layers 
        
        forward hook updates activation dictionary each time.
        """

        assert self.watched, "Please set which layers to watch first using set_watch()."
        # hook's outputs aren't activation values.

        for l_name in self.watched_layers:
            layer = self.model_params.get_layer(l_name)
            layer.register_forward_hook(self._get_outputs_hook(l_name))

        print("[TB-Hist] Outputs forward hooks registered.")
        self.outputs_hooked = True

    def _get_outputs_hook(self, l_name):
        def hook(module, inputs, outputs):
            self.outputs_cache[l_name] = outputs.detach()
        return hook
    
    def plot_outputs_hist(self, interval):
        """ plot activations stored in the activation cache """
        
        assert self.watched, "Please set which layers to watch first using set_watch()."
        assert self.outputs_hooked, "Please register forward hooks first using register_activation_hooks()"
        
        activation = self.outputs_cache
        for l_name in self.watched_layers:
            self.writer.add_histogram(tag=l_name+".outputs",
                                    values=activation[l_name],
                                    global_step=interval)

if __name__ == "__main__":
    from transformers import BertModel, BertConfig

    class BERTNli(BertModel):
        """ Fine-Tuned BERT model for natural language inference."""
        
        def __init__(self, config):
            super().__init__(config)
            self.fc1 = nn.Linear(768, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 1)
            
        
        def forward(self,
                input_ids=None,
                token_type_ids=None):
            x = super(BERTNli, self).forward(input_ids=input_ids, token_type_ids=token_type_ids)
            (_, pooled) = x    # see huggingface's doc.
            x = F.relu(self.fc1(pooled))
            x = F.relu(self.fc2(x))
            x = torch.sigmoid(self.fc3(x))
            return x

    # Specify BERT config
    # Load pre-trained model weights.
    model = BERTNli(BertConfig(max_length=64)).from_pretrained('bert-base-uncased')  # BertConfig returns 12 layers on default.
    params = ModelParams(model)

    print(params.params_dict.get('fc3.bias'))
    print(params.grads_dict.get('fc3.bias'))


