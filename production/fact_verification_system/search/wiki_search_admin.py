""" Search for relevant sentences in Elastic Search database

This module provides the abstraction for the fact_verification_system.

"""
from elasticsearch import Elasticsearch
import yaml

class WikiSearchAdmin(object):
    _es = None
    _indices = ['wiki']

    def __init__(self, config_path):
        super().__init__()
        # get host and port from config.yaml
        assert config_path.endswith('.yaml'), "[WSA] Configuration must be a yaml file."
        try:
            with open(config_path, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        except FileNotFoundError:
            print("[DBA] Config file not found.")
        except Exception as e:
            print("[DBA] {}".format(e))
        
        self._host = config['host']
        self._port = config['port']

        self._connect(self._host, self._port)


    @property
    def es(self):
        es = self.__class__._es
        if es is None:
            raise AttributeError("[Search] Elasticsearch instance must be established.")
        return es

    @property
    def host(self):
        return self._host

    @property
    def port(self):
        return self._port

    
    def _connect(self, host, port):
        self.__class__._es = Elasticsearch(hosts=[
            {'host': self._host, 'port': self._port}
        ])
        if not self._es.ping():
            raise ConnectionError("[Search] Elasticsearch connection \
                to {}:{} is not established.".format(self._host, self._port))
        else:
            print("[WSA] Connection to Elasticsearch established.")

    def health(self):
        return self.__class__._es.health(index=self.__class__._indices[0])

if __name__ == "__main__":
    wsa = WikiSearchAdmin(config_path='fact_verification_system/search/config.yaml')