"""
Database Context Manager for PostgreSQL.

requires a config.yaml file.
specifying host, port user, password and dbname.
"""
import yaml
import sqlalchemy

class Database(object):
    def __init__(self, config:str):
        super().__init__()
        # get host and port from config.yaml
        assert config.endswith('.yaml'), "[DB] Configuration must be a yaml file."
        try:
            with open(config, 'r') as f:
                postgres = yaml.load(f, Loader=yaml.FullLoader)
        except FileNotFoundError as fe:
            print("[DB] Config file not found.")
        except Exception as e:
            print("[DB] {}".format(e))

        self.host = postgres['host']
        self.port = postgres['port']
        self.user = postgres['user']
        self.password = postgres['password']
        self.dbname = postgres['dbname']

        self.conn = None
        self.cursor = None

    def connect(self):
        """ Connect to postgres DB with specified host and port
        
        Returns
            conn -- connection object
        """



if __name__ == "__main__":
    pass