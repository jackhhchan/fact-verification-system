""" Post requests to tensorflow serving model.

NOTE: Docker container tensorflow/serving must be running.
(more info at https://www.tensorflow.org/tfx/serving/docker)
"""

import json
import requests
from typing import Tuple, List

from fact_verification_system.classifier.pipeline.bert.preprocess import get_embeddings

# current model trained by vm.
max_seq_length = 64         
MODEL_NAME = "nli-bert-classifier"      # set when running docker container.
VERSION = 1                             # set as export path when saving model. (load_and_save.py)

class TFXPredict(object):

    def __init__(self,
        MODEL_NAME=MODEL_NAME,
        VERSION=VERSION):
        self.MODEL_NAME = MODEL_NAME    
        self.VERSION = str(VERSION)

        self.port = 8501
        self.max_seq_length = max_seq_length
        
        try:
            self._check_model_status()
        except ConnectionError as e:
            print("[TFX] Unable to connect to tensorflow serving on port {}".format(self.port))
            raise e

    @property
    def tfx_url(self):
        return "http://localhost:{}/v{}/models/{}".format(self.port, self.VERSION, self.MODEL_NAME)
     
    def post_predictions(self, bert_sents_list:List[Tuple[str]]) -> List[str]:
        """ Returns a list of predicted labels. """
        instances = list()
        for bert_sents in bert_sents_list:
            bert_dict = get_embeddings(bert_sents, self.max_seq_length, pythonList=True)
            instances.append(bert_dict)

        data = self._get_json_data(instances)
        headers = self._get_headers()

        json_res = requests.post(self.tfx_url, data=data, headers=headers)

        predictions = json.loads(json_res.text)['predictions']
        return list([self._get_label(pred[0]) for pred in predictions])


    def _check_model_status(self):

        json_res = requests.get(self.tfx_url)
        state = json.load(json_res)["model_version_status"][0]["state"]
        if not state == "AVAILABLE":
            raise ConnectionAbortedError(
        "[TFX] Model: {}, Version: {} is not available.".format(
                                                self.MODEL_NAME,
                                                self.VERSION))

    def _get_json_data(self, instances:list):
        data = json.dumps(
            {"signature_name": "serving_default",
            "instances": instances}
        )
        return data

    def _get_headers(self):
        return {"content_type": "application/json"}

    def _get_label(self, pred:float) -> str:
        if pred == 0.5 or pred > 0.5:
            return "SUPPORTS"
        else:
            return "REFUTES"
