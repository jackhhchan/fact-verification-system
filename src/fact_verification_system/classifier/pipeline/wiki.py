"""
This is a custom class specifically for the wiki-text-file dataset.

Used as input pi
"""
import json
from postgres.db_query import DatabaseQuery as dbq
from postgres.db_admin import DatabaseAdmin

class WikiGenerator(object):
    def __init__(self):
        pass

    # Load train.json
    def load(self, path='../../dataset/train.json'):
        with open(path, 'r') as j:
            train_json = json.load(j)
        yield train_json

    # data generator helper functions
    def get_sentence(self, evidence, session):
        page_id = evidence[0]
        sent_idx = evidence[1]
        return dbq.query_sentence(session, page_id, sent_idx)

    def data_generator(self, train_json, session):
        """ Generates data in [claim, sent, label]"""
        for t in train_json:
            claim = t.get('claim')
            label = t.get('label')
            evidences = t.get('evidence')
            for ev in evidences:
                sent = get_sentence(ev, session)
                yield claim, sent, label


    # connect to database
    dba = DatabaseAdmin('postgres/config.yaml')
    dba.connect()
    with dba.session() as session:
        for claim, sent, label in self.data_generator(train_json, session):
            pass
            # TODO: preprocess the data.
            # yield preprocessed