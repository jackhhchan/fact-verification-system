from sqlalchemy.orm.session import Session

class DatabaseQuery(object):
    """ Handles Database Queries

    Not coupled with database session.
    All static functions
    """

    @staticmethod
    def query_template(session:Session, **args) -> None:
        return session.query()