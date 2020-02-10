from elasticsearch.helpers import parallel_bulk
from collections import deque
import os
import multiprocessing
from typing import List

from logger.logger import Logger, Modes
from fact_verification_system.search.wiki_search_admin import WikiSearchAdmin

def seed():
    index_name = 'wiki'
    
    config = 'fact_verification_system/search/config.yaml'
    es = WikiSearchAdmin(config).es

    # ES 7.0 has removed mapping types, API calls are now Typeless
    print("Creating new index: {}...".format(index_name))
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
    es.indices.create(index=index_name)
    
    print("Seeding newly created index: {}...".format(index_name))
    cpu_count = multiprocessing.cpu_count()
    
    # deque(parallel_bulk(
    # client=wsa.es,
    # actions=_iter_wiki_data_for_es(index_name),
    # thread_count=cpu_count
    # ), maxlen=0)

    pb = parallel_bulk(
    client=es,
    actions=_iter_wiki_data_for_es(index_name),
    thread_count=cpu_count
    )

    for success, info in pb:
        if not success:
            print("A document failed: {}".format(info))

    es.indices.refresh()

    print("Bulk insert complete.")



def _iter_wiki_data_for_es(index_name):
    """ Checked and formatted wiki data to be inserted into ES"""
    for parsed in _iter_wiki_data():
        if _check(parsed):
            yield {
            "_index":index_name,
            'page_id': parsed[0],
            'sent_idx': parsed[1],
            'sentence': parsed[2],
        }

def _check(parsed:List[str]) -> bool:
    """ [Async] Final type cast check before inserting data to DB. """
    try:
        str(parsed[0])
        int(parsed[1])
        str(parsed[2])
        return True
    except ValueError:
        Logger.log("Unable to parse sent_idx in {}".format(parsed), mode=Modes.es_insert)
        return False
    except Exception:
        Logger.log("General parse error in {}".format(parsed), mode=Modes.es_insert)
        return False


def _iter_wiki_data():
    """ Generator for wiki data 
    
    Reads data from wiki-text files and generate formatted list.
    """
    path = '../dataset/wiki-pages-text'
    print("Parsing text files in {}...".format(path))
    wiki_fnames = [f for f in os.listdir(path) if f.endswith('.txt')]
    for wiki in wiki_fnames:
        with open("{}/{}".format(path, wiki), 'r') as f:
            for line in f:
                yield _parse_txt(line)


def _parse_txt(line:str) -> list:
    """ Parses each line in all the wiki pages texts. 
    
    Args:
    line - RAW line in a wiki-pages-texts.

    Returns:
    List; [page_id, sent_idx, sent]
    """
    line = line.split()
    pg_id = line[0]
    sent_idx = line[1]
    sent = ' '.join(line[2:])
    return [pg_id, sent_idx, sent]


if __name__ == "__main__":
    seed()