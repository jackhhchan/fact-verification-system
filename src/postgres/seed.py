""" 
DATABASE SEED SCRIPT

parses wiki-text files and
insert into postgres docker container.
"""
import os
from typing import List
import asyncio

from logger.logger import Logger, Modes

async def main():
    """ [Async] Insert to database. """
    await asyncio.gather(*[insert(parsed) for parsed in wiki_data()])

#### Database ####

async def insert(parsed):
    check = await check(parsed)
    if check:
        # insert into DB.

async def check(parsed:List[str]) -> bool:
    """ [Async] Final type cast check before inserting data to DB. """
    try:
        pg_id = str(parsed[0])
        sent_idx = int(parsed[1])
        sent = str(parsed[2])
        return True
    except ValueError as ve:
        await Logger.log("Unable to parse sent_idx in {}".format(parsed), mode=Modes.postgres_insert)
        return False
    except Exception as e:
        await Logger.log("General parse error in {}".format(parsed), mode=Modes.postgres_insert)
        return False


#### Wiki Text Files ####

def parse_txt(line:str) -> List[str]:
    """ Parses each line in all the wiki pages texts. 
    
    Args:
    line - RAW line in a wiki-pages-texts.

    Returns:
    List; [page_id, sent_idx, sent]
    """
    try:
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
    wiki_fnames = [f for f in os.listdir(path) if f.endswith('.txt')]
    for wiki in wiki_fnames:
        with open("{}/{}".format(path, wiki), 'r') as f:
            for line in f:
                yield parse_txt(line)

            
if __name__ == "__main__":
    asyncio.run(main())