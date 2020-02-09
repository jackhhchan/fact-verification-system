""" Evaluate current search strategy using train.json
"""

import json
from tqdm import tqdm
import argparse

from fact_verification_system.search.wiki_search_admin import WikiSearchAdmin
from fact_verification_system.search.wiki_search_query import WikiSearchQuery as wsq

def eval(args):
    config = 'fact_verification_system/search/config.yaml'
    print("Running evaluation script for search engine...")
    print("Using elasticsearch config path: {}".format(config))

    total = 0
    total_matched = 0
    es = WikiSearchAdmin(config_path=config).es
    with open('../dataset/devset.json', 'r') as fp:     # NOTE: change to train.json when you receive the dataset
        train_json = json.load(fp)

        for i, data in tqdm(enumerate(train_json.values())):
            claim = data.get('claim')
            # get RETRIEVED (page_id, sent_idx) tuples from returned search results
            rel_pageid_sentidx = wsq.query(es, claim).get_page_id_sent_idx(limit=10)
            

            # get TRUE (page_id, sent_idx) tuples from devset.json
            true_pageid_sentidx = set([(ev[0], ev[1]) for ev in data.get('evidence')])

            matched = len(rel_pageid_sentidx.intersection(true_pageid_sentidx))
            total_matched += matched

            total += matched
            total += len(data.get('evidence'))

            if args.debug:
                print("retrieved:\n {}".format(rel_pageid_sentidx))
                print("true:\n {}".format(true_pageid_sentidx))
                if i >= args.debug:
                    print("Eval ended. (DEBUG MODE)")
                    return

    # recall = (retrieved & true) / true
    recall = float(total_matched) / float(total)
    print("Recall: {:.2f}".format(recall))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", 
                    help="debug mode, loop ends at 3rd iteration.",
                    type=int)
    args = parser.parse_args()
    eval(args)