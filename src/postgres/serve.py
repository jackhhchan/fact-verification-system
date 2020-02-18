from flask import Flask

app = Flask(__name__)

from flask import request, abort
from flask_restplus import Resource, Api, fields
import requests
import json

from postgres.db_admin import DatabaseAdmin
from postgres.db_query import DatabaseQuery as dbq

dba = DatabaseAdmin('postgres/config.yaml')
dba.connect()

api = Api(app)


@api.route("/health")
class Health(Resource):
    def get(self):
        return {'data': "Postgresql is active."}

        

@api.route("/query")
class Query(Resource):
    def post(self):
        if request.json:
            j_data = request.get_json()
            try:
                page_id = j_data['page_id']
                sent_idx = j_data['sent_idx']
            except Exception as e:
                print("[/query] No page_id or sent_idx in request.")
                raise e
            with dba.session() as sess:
                sent = dbq.query_sentence(sess, page_id, sent_idx)
            return {"data": {
                "sentence":sent
            }}
        else:
            return {"Error": "Request must contain json."}






if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5431, debug=True)