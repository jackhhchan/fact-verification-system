from sqlalchemy import Column, Integer, String
from sqlalchemy.types import SMALLINT

from . import Base


class Data(Base):
    __tablename__ = 'data'
    
    id = Column(Integer, primary_key=True)
    page_id = Column(String)
    sent_idx = Column(SMALLINT)
    sentence = Column(String)

    def __repr__(self):
        return "<Data(page_id={}, sent_id={}, sentence={})".format(
            self.page_id, self.sent_id, self.sentence
        )