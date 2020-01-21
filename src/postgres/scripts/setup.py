"""
Set up table structure for database
"""
from postgres.db_admin import DatabaseAdmin

def setup():
    # set up sqlalchemy config
    dba = DatabaseAdmin('postgres/config.yaml')
    dba.connect()
    with dba.session() as session:
        # set up database
        # 





if __name__ == "__main__":
    setup()