import json
from enum import Enum

from fact_verification_system.classifier.preprocess import preprocess
from fact_verification_system.classifier.models import textual_entailment
# from fact_verification_system.classifier.pipeline.wiki import WikiGenerator

## NOTE: Manually tune parameters in Hyperparams
class Hyperparams(Enum):
    BATCH_SIZE = 64
    EPOCHS = 10
    OPTIMIZER = 'adam'
    LOSS = 'binary_cross_entropy'
    METRICS = ['accuracy']

class BertParams(Enum):
    MAX_SEQ_LENGTH = 512



def main():
    # load training file.
    train_json = json.load('train.json')    

    # generate datasets
    ds = None       # generator

    print("Dataset preview:")
    print(next(iter(ds)))
    print(type(ds))


    model = textual_entailment.create_model(max_seq_length=BertParams.MAX_SEQ_LENGTH.value)
    model.compile(optimizer=Hyperparams.OPTIMIZER.value,
                loss=Hyperparams.LOSS.value,
                metrics=Hyperparams.METRICS.value)

    model.fit(x=ds.shuffle(1000).batch(Hyperparams.BATCH_SIZE.value),
            epochs=Hyperparams.EPOCHS.value)


if __name__ == "__main__":
    main()