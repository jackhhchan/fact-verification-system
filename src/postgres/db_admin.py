import yaml
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

class DatabaseAdmin(object):
    """ Handles Database Sessions

    Must be instantiated with a database config yaml file.
    Call connect to set up connection to database.
    """


    def __init__(self, config_path:str):
        super().__init__()
        # get host and port from config.yaml
        assert config_path.endswith('.yaml'), "[DBA] Configuration must be a yaml file."
        try:
            with open(config_path, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        except FileNotFoundError:
            print("[DBA] Config file not found.")
        except Exception as e:
            print("[DBA] {}".format(e))

        self._host = config['host']
        self._port = config['port']
        self._user = config['user']
        self._password = config['password']
        self._dbname = config['dbname']
        self._engine = None
    
    @property
    def host(self):
        return self._host
    @property
    def user(self):
        return self._user
    @property
    def dbname(self):
        return self._dbname
    @property
    def db_string(self):        
        #dialect+driver://username:password@host:port/database
        return "postgresql://{}:{}@{}:{}/{}".format(
                                                    self._user,
                                                    self._password,
                                                    self._host,
                                                    self._port,
                                                    self._dbname)
    @property
    def engine(self):
        if self._engine is None:
            raise AttributeError("[DBA] engine was never created.")
        return self._engine


    def connect(self):
        """Creates engine and session maker binding to database engine"""
        if self._engine is None:
            try:
                print(
                    "[DBA] Connecting to {}:{}".format(self._host, self._port)
                )
                self._engine = create_engine(self.db_string)
                self._sessionmaker = sessionmaker(bind=self._engine)
                print("[DBA] Connected to {}:{}/{}.".format(
                                                        self._host,
                                                        self._port,
                                                        self._dbname))
            except Exception as e:
                raise ImportError("[DBA] {}".format(e))
            

    @contextmanager
    def session(self):
        """ Binds session to engine """
        # check if engine & session exist.
        assert (self._engine is not None and self._sessionmaker is not None), \
            "[DBA] Engine and Sessionmaker has not been created. Call connect() to instantiate."

        session = self._sessionmaker()
        try:
            yield session
            # after with block completes:
            print("[DBA] session commiting...")
            session.commit()    
            print("[DBA] session commited.")
        except:
            session.rollback()
            raise Exception("[DBA] Session scope failed. Session is rolled back.")
        finally:
            session.close()



if __name__ == "__main__":
    dba = DatabaseAdmin('postgres/config.yaml')
    dba.connect()
    with dba.session() as s:
        print("running with block...")
    print("done.")
