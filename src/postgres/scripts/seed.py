""" 
DATABASE SEED SCRIPT

Prereqs:
    postgres/config.yaml -- file with postgres config
    postgres/scripts/setup.py is run to set up database.

What it does:
    parses wiki-text files from '../dataset/wiki-pages-text' and
    insert into postgres docker container (connected via postgres/config.yaml)
"""
import os
from typing import List
import asyncio
from time import time
from sqlalchemy import inspect

from logger.logger import Logger, Modes
from postgres.tables.wiki import Wiki
from postgres.db_admin import DatabaseAdmin

async def main():
    """ [Async] Insert to database. """
    dba = DatabaseAdmin('postgres/config.yaml')
    dba.connect()

    tables = inspect(dba.engine).get_table_names()
    if not 'wiki' in tables:
        raise IndexError("Index Wiki not found. Please run setup.py")

    path = '../dataset/wiki-pages-text'
    print("Parsing text files in {}...".format(path))
    wiki_fnames = [f for f in os.listdir(path) if f.endswith('.txt')]
    
    # split in half so half is commited to db first to clear RAM.
    split = int(len(wiki_fnames)/4)
    wiki_fnames_0 = wiki_fnames[:split]
    wiki_fnames_1 = wiki_fnames[split:2*split]
    wiki_fnames_2 = wiki_fnames[2*split:3*split]
    wiki_fnames_3 = wiki_fnames[3*split:]

    wiki_fnames_splits = [wiki_fnames_0, wiki_fnames_1, wiki_fnames_2, wiki_fnames_3]
    
    s = time()
    for wiki_fnames_split in wiki_fnames_splits:
        with dba.session() as session:
            await asyncio.gather(*[insert(parsed, session) for parsed in wiki_data(path, wiki_fnames_split)])
        
    print("Elapsed time: {}s".format(time()-s))

#### Database ####
async def check(parsed:List[str]) -> bool:
    """ [Async] Final type cast check before inserting data to DB. """
    try:
        str(parsed[0])
        int(parsed[1])
        str(parsed[2])
        return True
    except ValueError:
        await Logger.log_async("Unable to parse sent_idx in {}".format(parsed), 
                            mode=Modes.postgres_insert)
        return False
    except Exception:
        await Logger.log_async("General parse error in {}".format(parsed), 
                    mode=Modes.postgres_insert)
        return False

async def insert(parsed:List[str], session):
    checked = await check(parsed)
    if checked:
        # insert into DB.
        wiki_data = Wiki(
            page_id=str(parsed[0]),
            sent_idx=int(parsed[1]),
            sentence=str(parsed[2]),
            )
        session.add(wiki_data)



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



def wiki_data(path:str, wiki_fnames:list):
    """ Generator for wiki data 
    
    Reads data from wiki-text files and generate formatted list.
    """

    for wiki_fname in wiki_fnames:
        print("[POSTGRES-SEED] Inserting {} to postgres db...".format(wiki_fname))
        with open("{}/{}".format(path, wiki_fname), 'r') as f:
            for i, line in enumerate(f):
                i = i + 1
                if i % 10000 == 0:
                    print("[POSTGRES-SEED] {} lines parsed for {}.".format(i, wiki_fname))
                yield parse_txt(line)

            
if __name__ == "__main__":
    asyncio.run(main())
    # d = next(wiki_data())
    # print(d)