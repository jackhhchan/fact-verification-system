""" Set up script create wiki data index.

Parallel import is used.
"""
from elasticsearch.exceptions import RequestError

from fact_verification_system.search.wiki_search_admin import WikiSearchAdmin


def setup():
    print("Setting up elasticsearch database...")
    index_name = 'wiki'
    wsa = WikiSearchAdmin(host="0.0.0.0", port=9200)
    try:
        print("Creating index: {}".format(index_name))
        wsa.es.indices.create(index=index_name)
    except RequestError as e:
        print(e)
        print("[setup.py] Index: {} already exists.".format(index_name))
        return None

    except Exception as e:
        print("[setup.py] Error while setting up elasticsearch index.")
        raise e    




if __name__ == "__main__":
    setup()