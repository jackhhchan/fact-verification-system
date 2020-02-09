from enum import Enum
from elasticsearch.exceptions import NotFoundError
from elasticsearch_dsl import Search, Q
from elasticsearch import Elasticsearch

class Field(Enum):
    page_id = 'page_id'
    sent_idx = 'sent_idx'
    sentence = 'sentence'

class Analyzer(Enum):
    standard = 'tokenises with Unicode Text Segmentation, \
                removes punctuations, lowercase, \
                and supports removing stop words.'
    simple = "Tokenise and lowercase."
    stop = "Tokenise, lowercase and remove stop words."
    # more on 
    # https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-analyzers.html
    

class WikiSearchQuery(object):
    index = 'wiki'
    s = None

    def __init__(self, es:Elasticsearch):
        self._create_search(es)

    def _create_search(self, es:Elasticsearch):
        self.__class__.s = Search(using=es)


    @classmethod
    def query(cls, es, input_claim:str) -> list:         # NOTE: change return to list later
        """ Returns a list of relevant results. """
        try:
            cls.add_match_query(input_claim)
            # cls.add_multi_match_query(input_claim)
            res = cls.s.execute()   # sends request to ES
        except AttributeError as e:
            print("[WSQ] Please instantiate WikiSearchQuery object first.")
            raise e
        except Exception as e:
            print("[WSQ] Error with WikiSearchQuery.")
            raise e
        return list([hit for hit in res])
    
    @classmethod
    def add_match_query(cls, input_claim):
        kwargs = {
            Field.sentence.name: input_claim
        }
        cls.s.query('match', **kwargs)

    @classmethod
    def add_multi_match_query(cls, input_claim):
        q = Q("multi_match", 
            query=input_claim, 
            fields=[
                Field.sentence.name,
                Field.page_id.name
                ])
        cls.s.query(q)
