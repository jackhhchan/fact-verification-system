""" Evaluates sentence selection module using train.json

Did recall increase after sentence selection filtering is applied?
"""
import json
from tqdm import tqdm
import argparse

from fact_verification_system.sentence_selection.sentence_selection import SentenceSelection
from fact_verification_system.search.wiki_search_admin import WikiSearchAdmin
from fact_verification_system.search.wiki_search_query import WikiSearchQuery as wsq
# NOTE: requires ES scripts to be written.
def eval(args):
    config = 'fact_verification_system/search/config.yaml'
    ss = SentenceSelection()
    es = WikiSearchAdmin(config).es

    total = 0
    total_matched = 0

    with open('../dataset/devset.json', 'r') as fp:     # NOTE: change to train.json when you receive the dataset
        train_json = json.load(fp)

        for i, data in enumerate(train_json.values()):
            claim = data.get('claim')
            # get RETRIEVED (page_id, sent_idx) tuples from returned search results
            limit = 10
            res = wsq.query(es, claim)
            rel_pageid_sentidx = res.get_sentences(limit=limit)
            sentences = res.get_sentences(limit=limit)
            
            # sentence selection
            filtered_sentences = ss.filtered_sentences(claim, sentences)
            indices = (i for (i, sent) in filtered_sentences)
            

            # modified retrieved data
            filtered_rel = set(filter(lambda rel: rel[0] in indices, rel_pageid_sentidx))

            # get TRUE (page_id, sent_idx) tuples from devset.json
            true_pageid_sentidx = set([(ev[0], ev[1]) for ev in data.get('evidence')])

            matched = len(filtered_rel.intersection(true_pageid_sentidx))
            total_matched += matched

            total += matched
            total += len(data.get('evidence'))

            if args.debug:
                print("retrieved:\n{}".format(rel_pageid_sentidx))
                print("filtered_sentences:\n{}".format(filtered_sentences))
                print("filtered:\n{}".format(filtered_rel))
                print("true:\n{}".format(true_pageid_sentidx))
                if i >= args.debug - 1:
                    print("Eval ended. (DEBUG MODE)")
                    return

    # recall = (retrieved & true) / true
    recall = float(total_matched) / float(total)
    print("Recall: {:.2f}".format(recall))
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", 
                    help="debug mode, loop ends at x iteration.",
                    type=int)
    args = parser.parse_args()
    print(args.debug)
    eval(args)