"""
Naive bayes

Refresher:
P(class=A|example) = P(class=A) x P(feature 1 in class A) x P(feature 2 in class A) x ...
same for class = B
...

Classification = P(class=?|example) that is highest.

Caveat:
If feature X does not exist in class A, then it'll have a probability of 0. Hence we use an alpha of >= 1
alpha just adds 1 to any feature not in class A, but is in other classes.
"""

import pandas as pd
from sklearn.naive_bayes import MultinomialNB, GaussianNB

def main(path: str):
    df = pd.read_csv(path)


    m_nb = MultinomialNB()


if __name__ == '__main__':
    main()