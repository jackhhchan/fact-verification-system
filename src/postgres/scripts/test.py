import os
from typing import List

from logger.logger import Logger, Modes
from postgres.tables.wiki import Wiki
from postgres.db_admin import DatabaseAdmin

def seed():
    dba = DatabaseAdmin('postgres/config.yaml')
    dba.connect()
    with dba.session() as session:
        session.add(Wiki(
            page_id='fake_page_id',
            sent_idx=1,
            sentence='fake sentence'
        ))
        # for d in wiki_data():
        #     print(d)
        #     insert(d, session)
        #     import time
        #     time.sleep(30)
        #     break



def insert(parsed:List[str], session):
    checked = check(parsed)
    if checked:
        # insert into DB.
        data = Wiki(
            page_id=str(parsed[0]),
            sent_idx=int(parsed[1]),
            sentence=str(parsed[2]),
            )
        print(data)
        session.add(data)


def check(parsed:List[str]) -> bool:
    """ [Async] Final type cast check before inserting data to DB. """
    try:
        str(parsed[0])
        int(parsed[1])
        str(parsed[2])
        return True
    except ValueError:
        print("Check is not passed. {}".format(parsed[0]))
        return False
    except Exception:
        return False

#### Wiki Text Files ####

def parse_txt(line:str) -> List[str]:
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



def wiki_data():
    """ Generator for wiki data 
    
    Reads data from wiki-text files and generate formatted list.
    """
    path = '../dataset/wiki-pages-text'
    print("Parsing text files in {}...".format(path))
    wiki_fnames = [f for f in os.listdir(path) if f.endswith('.txt')]
    for wiki_fname in wiki_fnames:
        print("[POSTGRES-SEED] Inserting {} to postgres db...".format(wiki_fname))
        with open("{}/{}".format(path, wiki_fname), 'r') as f:
            for i, line in enumerate(f):
                i = i + 1
                if i % 10000 == 0:
                    print("[POSTGRES-SEED] {} lines parsed for {}.".format(i, wiki_fname))
                yield parse_txt(line)


if __name__ == "__main__":
    seed()