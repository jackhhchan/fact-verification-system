from sqlalchemy.orm.session import Session
from sqlalchemy.orm.exc import NoResultFound, MultipleResultsFound

from postgres.tables.wiki import Wiki

class DatabaseQuery(object):
    """ Handles Database Queries

    Not coupled with database session.
    All static functions
    """

    @staticmethod
    def query_template(session:Session, **args) -> None:
        return session.query()

    @staticmethod
    def query_sentence(session: Session, page_id:str, sent_idx:int) -> str:
        """ Returns a single sentence for the specific page_id and sent_idx. """
        try:
            ev =  session.query(Wiki).\
                filter(Wiki.page_id==page_id).\
                filter(Wiki.sent_idx==sent_idx).one()
            return ev.sentence
        except NoResultFound:
            print("[DBQ] No results found for page_id: {}, sent_idx: {}".format(
                page_id, 
                sent_idx
            ))
        except MultipleResultsFound:
            print("[DBQ] Multiple results found for page_id: {}, sent_idx: {}".format(
                page_id, 
                sent_idx
            ))
        except Exception as e:
            print("[DBQ] Query error: {}".format(e))