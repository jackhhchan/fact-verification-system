"""
Set up table structure for database
"""
from postgres.db_admin import DatabaseAdmin
from postgres.db_schema import DatabaseSchema

def setup():
    """ Sets up database with the appropriate schemas
    """
    # connect to db
    dba = DatabaseAdmin('postgres/config.yaml')
    dba.connect()

    # set up schemas
    dbs = DatabaseSchema()
    dbs.create_schemas(dba.engine)

if __name__ == "__main__":
    setup()