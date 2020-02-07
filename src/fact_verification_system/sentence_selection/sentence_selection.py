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
        return ('PERSON')       #TODO: there are more to be added

    def filtered_sentences(self, input_claim, sentences:List[str]) -> List[str]:
        """ Returns list of filtered sentences

        filtering conditions:
            - at least one NER tag matches
        """
        return filter(
                    lambda sent: self._match_conditions(input_claim, sent),
                    sentences)

    # NOTE: match conditions is extensible by adding more function that returns boolean.
    def _match_conditions(self, claim, sent) -> bool:
        doc_claim = self.nlp(claim)
        doc_sent = self.nlp(sent)

        # NOTE: conditions is extensible by adding more Condition subclasses. 
        conditions = [
            OneTagMatch(doc_claim, doc_sent)
        ]

        for cond in conditions:
            if not isinstance(cond, Condition):
                raise AttributeError("[SS] conditions in list must be an instances of class Condition.")
            if cond.meet_condition() is False:
                return False
        return True
        

    def _one_tag_match(self, doc_1, doc_2) -> bool:
        doc_1_tags = self._get_NER_tags(doc_1)
        doc_2_tags = self._get_NER_tags(doc_2)
        return len(doc_1_tags.intersection(doc_2_tags)) > 0


    def _get_NER_tags(self, doc:spacy.tokens.doc.Doc) -> set:
        """ Uses spaCy's NER tagger to return NER tags

        Return:
            A set of NER tags.
        
        Arg:
        sent -- sentence returned by ES
        """
        return set([e.label_ for e in doc.ents])



class Condition(object):
    def __init__(self, doc_1, doc_2):
        self.doc_1 = doc_1
        self.doc_2 = doc_2
    
    def meet_condition(self, *args) -> bool:
        raise NotImplementedError("[SS] Must implement meet_condition().")

class OneTagMatch(Condition):
    def meet_condition(self, *args):
        try:
            _get_NER_tags = args[0]
        except IndexError:
            raise IndexError(
                    "[SS] OneTagMatch condition must have _get_NER_tags() \
                    method as first argument.")
        if not callable(_get_NER_tags):
            raise ValueError('[SS] _get_NER_tags must be a function.')
        doc_1_tags = _get_NER_tags(self.doc_1)
        doc_2_tags = _get_NER_tags(self.doc_2)
        return len(doc_1_tags.intersection(doc_2_tags)) > 0