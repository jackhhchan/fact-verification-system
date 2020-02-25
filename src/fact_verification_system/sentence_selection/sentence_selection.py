from typing import List
import spacy

class SentenceSelection(object):
    def __init__(self):
        try:
            spacy_model = "en_core_web_sm"
            self.nlp = spacy.load(spacy_model)
        except:
            raise ImportError(
                "[SS] Spacy unable to load {}.\n\
                    exec in your environment: `python -m  spacy download {}"\
                    .format(spacy_model, spacy_model))



    @property
    def spacy_all_tags(self):
        return ('PERSON', 'more')       #TODO: there are more to be added

    def filtered_sentences(self, input_claim, sentences:List[str]) -> List[tuple]:
        """ Returns list of filtered sentences with index

        filtering conditions:
            - at least one NER tag matches
        """
        filtered = list(filter(
                    lambda x: self._match_conditions(input_claim, x[1]),        #x[0] is an index
                    enumerate(sentences)))

        # num_filtered = len(sentences) - len(filtered)
        # print("[SS] Filtered {} sentences.".format(num_filtered))
        return filtered

    def _match_conditions(self, claim, sent) -> bool:
        doc_claim = self.nlp(claim)
        doc_sent = self.nlp(sent)

        # NOTE: conditions is extensible by adding more Condition subclasses. 
        conditions = [
            OneTagMatch(doc_claim, doc_sent),
            SentenceMinNumTokens(None, doc_sent)
        ]

        for cond in conditions:
            if not isinstance(cond, Condition):
                raise AttributeError("[SS] conditions in list must be an instances of class Condition.")
            if cond.meet_condition() is False:
                return False
        return True






class Condition(object):
    def __init__(self, doc_1: spacy.tokens.doc.Doc, doc_2: spacy.tokens.doc.Doc):
        self.doc_1 = doc_1      # input_claim
        self.doc_2 = doc_2      # returned result from elastic search.
    
    def meet_condition(self) -> bool:
        raise NotImplementedError("[SS] Must implement meet_condition().")

class OneTagMatch(Condition):
    def meet_condition(self):
        if not callable(_get_NER_tags):
            raise ValueError('[SS] _get_NER_tags must be a function.')
        doc_1_tags = _get_NER_tags(self.doc_1)
        doc_2_tags = _get_NER_tags(self.doc_2)

        if len(doc_1_tags) == 0:
            return True
        else:
            return len(doc_1_tags.intersection(doc_2_tags)) >= 1

class SentenceMinNumTokens(Condition):
    """ Match results satisfying minimum number of tokens that constitute a sentence. """
    def meet_condition(self):
        # tokenize docs, check length.
        doc_filtered_punct = list(filter(lambda t: not t.is_punct, self.doc_2))
        return len(doc_filtered_punct) >= 3


def _get_NER_tags(doc:spacy.tokens.doc.Doc) -> set:
    """ Uses spaCy's NER tagger to return NER tags

    Return:
        A set of NER tags.
    
    Arg:
    sent -- sentence returned by ES
    """
    return set([e.label_ for e in doc.ents])