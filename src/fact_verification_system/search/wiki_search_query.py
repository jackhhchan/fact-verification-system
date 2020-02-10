from enum import Enum
from typing import Set, List
from elasticsearch.exceptions import NotFoundError
from elasticsearch_dsl import Search

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

    def __init__(self):
        pass


    @classmethod
    def query(cls, es, input_claim:str) -> dict:         # NOTE: change return to list later
        """ Returns a list of relevant results. """
        try:
            res = es.search(
                index=cls.index,
                body=cls._format_full_text_multi_match(input_claim)
            )
        except NotFoundError as e:
            print("[WSQ] Index {} does not exist.".format(cls.index))
            raise e
            
        return Results(res)


    @staticmethod
    def _format_boolean_query(token):
        """ DSL format for exact match with token 
        
        Uses Boolean query DSL format.
        """
        return {
            "query": {
                "bool": {
                    "must": [
                        {"match": {Field.sentence.name: token}}
                    ]
                }
            }
        }
    
    @staticmethod
    def _format_full_text_match(query):
        """ DSL format for search with query matching 'sentence' field
        
        Query is analyzed before it is queried.
        """
        return {
            "query": {
                "match": {
                    Field.sentence.name: {
                        "query": query
                    }
                }
            }
        }

    @staticmethod
    def _format_full_text_multi_match(query, page_id_boost:int=1):
        """ DSL for search with query matching both 'page_id' and 'sentence' fields.
        
        Args:
        page_id_boost (optional)
            - how many times is the field 'page_id' more important than 'sentence'.
        """
        return {
        "query": {
            "multi_match": {
                "query": query,
                "fields": [
                    "{}^{}".format(Field.page_id.name, page_id_boost),
                    Field.sentence.name
                    ]
                }
            }
        }


class Results(object):
    def __init__(self, results):
        assert type(results) is dict,\
            "Result object must be raw json returned by Elasticsearch query."
        self._results = results

    @property
    def results(self):
        return self._results

    def get_hits(self, limit:int=None)->list:
        try:
            hits = self._results['hits']['hits']
        except Exception as e:
            print("[WSQ] Result must be the raw json format returned by Elasticsearch.")
            raise e
        
        if limit is None:
            return hits
        else:
            try:
                hits = hits[:limit]
                return hits
            except IndexError as e:
                print("[WSQ] Hits limit specified is more than hits returned.")
                return hits

    def get_page_id_sent_idx(self, limit:int=None)-> Set[tuple]:
        hits = self.get_hits(limit)
        return set([(hit['_source']['page_id'], hit['_source']['sent_idx']) for hit in hits])

    def get_sentences(self, limit:int=None)->List[str]:
        hits = self.get_hits(limit)
        return list([hit['_source']['sentence'] for hit in hits])


    def __str__(self):
        return str(self._results)