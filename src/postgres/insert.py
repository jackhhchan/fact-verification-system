""" 
DATABASE INSERT SCRIPT

parses wiki-text files and
insert into postgres docker container.
"""
import os
from typing import List
import asyncio

from logger.logger import Logger, Modes

async def insert():
    # pull from wiki data generator
    # handle & log errors using logger
    print(wiki_data())
    for i, data in enumerate(wiki_data()):
        print(data)
        if i == 10:
            break
    await asyncio.gather(Logger.log('hello', mode=Modes.postgres_setup))


def parse(line:str) -> List[str]:
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
    """ Generator for wiki data """
    path = '../dataset/wiki-pages-text'
    wiki_fnames = [f for f in os.listdir(path) if f.endswith('.txt')]
    for wiki in wiki_fnames:
        with open("{}/{}".format(path, wiki), 'r') as f:
            for line in f:
                # logging
                yield parse(line)

            

if __name__ == "__main__":
    asyncio.run(insert())