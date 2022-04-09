from fact_verification_system.search.wiki_search_admin import WikiSearchAdmin
from fact_verification_system.search.wiki_search_query import WikiSearchQuery as wsq

import json
import pprint
pp = pprint.PrettyPrinter(indent=2)



def main():
    wsa = WikiSearchAdmin(config_path='fact_verification_system/search/config.yaml')

    # mock_input_claim = "Robert Downey jr is famous for playing Ironman."
    mock_input_claim = "Wildcats"


    res = wsq.query(wsa.es, mock_input_claim)
    # pp.pprint(res.results)
    pp.pprint(res.get_hits(limit=10))

def seed():

    es = WikiSearchAdmin(config_path='fact_verification_system/search/config.yaml').es
    print(es)
    print(dir(es))
    print(es.ping)
    print(es.info())
    # print(es.indices.exists('wiki'))


if __name__ == "__main__":
    # main()
    seed()