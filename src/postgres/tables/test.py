from postgresql.tables.data import Data



d = Data(
    page_id='page',
    sent_id=1,
    sentence='sentence'
)

print(d)
print(d.page_id)
print(str(d.id))

from . import Base

print(type(Base))