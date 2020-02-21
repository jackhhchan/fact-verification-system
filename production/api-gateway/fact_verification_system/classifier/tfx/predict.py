""" Post requests to tensorflow serving model.

NOTE: Docker container tensorflow/serving must be running.
(more info at https://www.tensorflow.org/tfx/serving/docker)
"""

import json
import requests
from typing import Tuple, List, Dict

from fact_verification_system.classifier.pipeline.bert.preprocess import get_embeddings

# current model trained by vm.
max_seq_length = 64         
MODEL_NAME = "nli-bert"                 # set when running docker container.
VERSION = 1                             # set as export path when saving model. (load_and_save.py)

class TFXPredict(object):

    def __init__(self,
        MODEL_NAME=MODEL_NAME,
        VERSION=VERSION):
        self.MODEL_NAME = MODEL_NAME    
        self.VERSION = str(VERSION)

        self.port = 8080                        # cloud run default port
        self.max_seq_length = max_seq_length
        
        try:
            self._check_model_status()
        except ConnectionError as e:
            print("[TFX] Unable to connect to tensorflow serving on port {}".format(self.port))
            raise e

    @property
    def tfx_url(self):
        # return "http://localhost:{}/v{}/models/{}".format(self.port, self.VERSION, self.MODEL_NAME)
        return "https://tfx-ddzqcpwcwq-an.a.run.app/v{}/models/{}".format(self.VERSION, self.MODEL_NAME)
     
    def post_predictions(self, bert_sents_list:List[Tuple[str]]) -> Dict[int, str]:
        """ Returns a list of predicted labels. """
        instances = list()
        for bert_sents in bert_sents_list:
            bert_dict = get_embeddings(bert_sents, self.max_seq_length, pythonList=True)
            instances.append(bert_dict)

        data = self._get_json_data(instances)
        headers = self._get_headers()

        tfx_url_predict = self.tfx_url + ":predict"

        json_res = requests.post(tfx_url_predict, data=data, headers=headers)

        if json_res.status_code == 200:
            predictions = json_res.json()['predictions']
            return dict({i: self._get_label(pred[0]) for (i, pred) in enumerate(predictions)})
            # return list([self._get_label(pred[0]) for pred in predictions])
        else:
            return json_res.json()


    def _check_model_status(self):

        json_res = requests.get(self.tfx_url)
        state = json.loads(json_res.text)["model_version_status"][0]["state"]
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
