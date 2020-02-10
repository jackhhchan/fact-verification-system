from . import Base
from postgres.tables.wiki import *

from sqlalchemy.engine import Engine

class DatabaseSchema(object):
    def __init__(self):
        super().__init__()

    @staticmethod
    def create_schemas(engine: Engine) -> None:
        # create schemas using engine's meta data.
        try:
            Base.metadata.create_all(engine)
            print("[DBS] Info:\n[DBS]{}".format(Base.metadata.tables))
            print("[DBS] Schemas created.")
        except Exception as e:
            raise Exception("[DBS] Error: {}".format(e))