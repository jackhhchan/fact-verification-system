""" REST request interface for the Fact Verification System
"""

from flask import Flask

app = Flask(__name__)

from flask import request, abort, Response
from flask_restplus import Resource, Api, fields

import pprint
pp = pprint.PrettyPrinter(indent=2).pprint

from fact_verification_system.search.wiki_search_admin import WikiSearchAdmin
from fact_verification_system.search.wiki_search_query import WikiSearchQuery as wsq
from fact_verification_system.sentence_selection.sentence_selection import SentenceSelection
from fact_verification_system.classifier.tfx.predict import TFXPredict


api = Api(app)

config = 'fact_verification_system/search/config.yaml'
es = WikiSearchAdmin(config).es
ss = SentenceSelection()

tfxp = TFXPredict()

@api.route('/health')
class Health(Resource):
    def get(self):
        # TODO: send a query to es, sentence selection then classifier.
        return formattedResponse(None, "Fact Verification System is active.")

@api.route('/evidence')
class Evidence(Resource):
    def post(self):
        if request.json:
            req = request.json
            claim = req['data']['claim']
            limit = req.get('limit', 10)
            
            ## send query to search engine
            print("Querying elasticsearch...")
            res = wsq.query(es, claim)
            hits = res.get_hits(limit=limit)
            sentences = res.get_sentences(limit=limit)
            pp(sentences)
            
            ## sentence selection
            # filtered sentences
            filtered_sentences = ss.filtered_sentences(claim, sentences)
            print("Filtered sentences:")
            pp(filtered_sentences)
            # filtered indices
            filtered_indices = [i for i in range(0, limit)]
            for (i, _) in filtered_sentences:
                filtered_indices.remove(i) 
            # print("Filtered indices:")
            # pp(filtered_indices)
            ## classifier
            # send claim and sentence pairs to classifier
            filtered_sentences = list([sent for (_, sent) in filtered_sentences])
            classifier_inputs = map(lambda sent: (claim, sent), filtered_sentences)

            # make prediction
            print("Classifying filtered sentences...")
            preds =  tfxp.post_predictions(classifier_inputs)
            pp(preds)

            preds_sentences = list(zip(filtered_sentences, preds.values()))
            return formattedResponse(meta=None, data=preds_sentences)
        else:
            message = ("Request must contain json. "
                 "'data' field should contain the claim string.")
            abort(400, message)

# add CORS headers before sending response back
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response



## HELPERS ##
def formattedResponse(meta, data):
    return {
        'meta': meta,
        'data': data
    }

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)