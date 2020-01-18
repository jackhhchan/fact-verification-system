import time
import os
from enum import Enum

import aiofiles
"""
Simple Logger

Logs to txt file.
"""
class Modes(Enum):
    """ Log Mode : File Path """
    postgres_insert = 'postgres_insert.txt'

    @classmethod
    def has_value(cls_, value) -> bool:
        return value in cls_.__members__
    @classmethod
    def is_type(cls_, value) -> bool:
        return type(value) == Modes

class Logger(object):
    def __init__(self):
        super().__init__()

    @staticmethod
    async def log(message:str, mode:Modes='') -> None:
        """ [Async] Logs error message in designated txt files. 
        
        Inputs:
            message -- message to be logged
            mode -- the designated text file to be logged in.
                    postgres_setup, 

        Side Effects:
            messages are timestamped automatically.
            fallback and logs to logs_temp.txt if invalid mode selected
        """
        fdir = "logs"
        fpath = ""

        if not Modes.is_type(mode):
            fpath = '{}/{}'.format(fdir, 'logs_temp.txt')
            print("Invalid logger mode. Logs now stored at {}\nMessage: {}".format(fpath, message))
        else:
            fpath = '{}/{}'.format(fdir, mode.value)

        if fpath.startswith("{}/".format(fdir)) and not os.path.isdir(fdir):
            os.mkdir(fdir)

        async with aiofiles.open(fpath, 'a') as f:
            await f.write("{}  {}\n".format(Logger._get_timestamp(), message))

    @staticmethod
    def _get_timestamp() -> str:
        return time.strftime("%c", time.gmtime())
        