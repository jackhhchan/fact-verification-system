""" Send a post request to tensorflow serving model.

NOTE: Docker container tensorflow/serving must be running.
(more info at https://www.tensorflow.org/tfx/serving/docker)
"""

import json
import requests

from fact_verification_system.classifier.pipeline.bert.preprocess import get_embeddings

def main():
    # mock data
    mock_claim = 'robert downey junior is iron man.'
    mock_evidence = "robert downey junior is a plumber, accountant and a professional dancer"
    # mock_evidence = "robert downey junior is definitely not iron man, that's crazy!."
    bert_sents = (mock_claim, mock_evidence)
    # print("bert_sents: {}".format(bert_sents))
    
    # convert mock data to input embeddings
    max_seq_length = 64
    bert_dict = get_embeddings(bert_sents, max_seq_length, pythonList=True)
    # print("bert_dict: {}".format(bert_dict))

    instances = [bert_dict]

    # format into json to send post request
    data = json.dumps(
        {"signature_name": "serving_default",
        "instances": instances}    # tuple of dicts
    )
    headers = {"content-type": "application/json"}

    tf_serving_url = "http://localhost:8501/v1/models/nli-bert:predict"
    # tf_serving_url = "https://tfx-ddzqcpwcwq-an.a.run.app/v1/models/nli-bert:predict"
    # print("Sending to {}...\ndata: {}".format(tf_serving_url, data))
    
    
    json_res = requests.post(tf_serving_url, data=data, headers=headers)
    # print("json_res: {}".format(json_res))

    predictions = json.loads(json_res.text)['predictions']

    print(predictions)

    for input_, pred in zip([bert_sents], predictions):
        print("{} : {}".format(input_, get_label(pred[0])))

def get_label(pred:float) -> str:
    if pred == 0.5 or pred > 0.5:
        return "SUPPORTS"
    else:
        return "REFUTES"


if __name__ == "__main__":
    main()