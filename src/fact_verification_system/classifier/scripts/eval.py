# Predict using the Textual Entailment model

from fact_verification_system.classifier.scripts.train import _extract, _parse_and_transform
from fact_verification_system.classifier.models import textual_entailment as te
import multiprocessing as mp
from tensorflow.keras.optimizers import SGD

def main():
    suffix_dev = "devset_64_balanced.tfrecord"
    file_pattern = "../dataset/tfrecords/" + suffix_dev

    num_cpus = mp.cpu_count()

    # extract data
    ds_val = _extract(file_pattern)
    ds_val = ds_val.map(_parse_and_transform, num_parallel_calls=num_cpus)

    # modelling
    model = te.create_bert_model(max_seq_length=64)
    model.load_weights("../trained_models/desktop/10000samples/model_weights.hdf5")

    model.summary()

    model.compile(optimizer=SGD(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy'])

    # evaluate
    results = model.evaluate(ds_val.batch(8))
    print('test loss, test acc:', results)



if __name__ == "__main__":
    main()