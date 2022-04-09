import pytest

from pytorch_tutorial.fvs.transforms import BertInputEmbeds, Truncate, BertSpecialTokens

def test_segments():
    max_seq_lengths = [3, 5, 7, 9, 11]
    for max_seq_length in max_seq_lengths:
        bie = BertInputEmbeds(max_seq_length)
        for i in range(100, 1):
            x = ["hello"]*i
            segments = bie._get_segments(x)
            assert len(segments) == max_seq_length

def test_input_ids():
    max_seq_lengths = [3, 5, 7, 9, 11]
    for max_seq_length in max_seq_lengths:
        bie = BertInputEmbeds(max_seq_length)
        for i in range(100, 1):
            x = ["hello"]*i
            ids = bie._get_ids(x)
            assert len(ids) == max_seq_length


def test_truncate():
    max_length = 5
    trun = Truncate(max_length)
    sample = {}
    sample['data'] = (['hi']*(max_length+1), ['th']*(max_length+1))
    truncated = trun(sample)
    assert len(truncated['data'][0]) <= max_length-3
    assert len(truncated['data'][1]) <= max_length-3

    
def test_special_tokens():

    max_length = 5
    trun = Truncate(max_length)
    sample = {}
    sample['data'] = (['hi']*(max_length+1), ['th']*(max_length+1))
    assert len(sample['data'][0] ) <= max_length + 1
    truncated = trun(sample)
    st = BertSpecialTokens()
    stTokens = st(truncated)
    assert len(stTokens) <= max_length
