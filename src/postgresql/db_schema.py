class DatabaseSchema(object):
    def __init__(self):
        super().__init__()

    @staticmethod
    def create_schemas(engine, meta):
        # create schemas using engine's meta data.
        pass