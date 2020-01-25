"""
Set up table structure for database
"""
from postgres.db_admin import DatabaseAdmin
from .. import Base
from postgres.tables import *

def setup():
    """ Sets up database with the appropriate schemas
    """
    # set up sqlalchemy config
    dba = DatabaseAdmin('postgres/config.yaml')
    dba.connect()
    Base.metadata.create_all(dba.engine)

if __name__ == "__main__":
    setup()