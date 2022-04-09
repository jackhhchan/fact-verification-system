from sqlalchemy import Column, Integer, String
from sqlalchemy.types import SMALLINT

from .. import Base


class Wiki(Base):
    __tablename__ = 'wiki'
    
    id = Column(Integer, primary_key=True)
    page_id = Column(String)
    sent_idx = Column(SMALLINT)
    sentence = Column(String)

    def __repr__(self):
        return "<Wiki(page_id={}, sent_id={}, sentence={})".format(
            self.page_id, self.sent_idx, self.sentence
        )