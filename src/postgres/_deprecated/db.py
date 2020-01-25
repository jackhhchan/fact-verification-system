"""
Database Context Manager for PostgreSQL.

requires a config.yaml file.
specifying host, port user, password and dbname.
"""
import yaml
import psycopg2

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
    
    # Context Manager magic methods
    def __enter__(self):
        """ Connects to database and return the cursor. """
        self.conn = self._connect()
        self.cursor = self.conn.cursor()
        return self.cursor

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """ Commit and closes connection to database. """
        print("Committing to db...")
        self.conn.commit()
        self.cursor.close()
        self.conn.close()

    def _connect(self):
        """ Connect to postgres DB with specified host and port
        
        Returns
            conn -- connection object
        """
        try:
            conn = psycopg2.connect(dbname=self.dbname,
                                    user=self.user,
                                    password=self.password,
                                    host=self.host,
                                    port=self.port)
            return conn
        except psycopg2.OperationalError as e:
            print(e)



if __name__ == "__main__":
    with Database('postgres/config.yaml') as cur:   
        cur.execute("CREATE TABLE test (id serial PRIMARY KEY, num integer, data varchar);")
        cur.execute("INSERT INTO test (num, data) VALUES (%s, %s);", (100, "abc'def"))
        cur.execute("SELECT * FROM test;")
        res = cur.fetchone()
        print(res)
        cur.execute("DROP TABLE test;")