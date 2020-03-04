# Fact Verification System

The Fact Verification System verify claims using from a subset of information from Wikipedia.

The system consist of 3 components:
  - Search (uses BM25 algorithm from Elasticsearch)
  - Sentence Selection (uses NER from SpaCy)
  - Natural Language Inference (uses a fine-tuned BERT model trained on Tensorflow 2.0)

### To run:
To run any module or scripts:
```
cd src/
python -m folder.module
```

## src/postgres
#### database storing raw dataset files
#### 


## src/fact_verification_system/classifier
#### classifier to be run on its own container.

## src/fact_verification_system/search
#### search engine to retrieve relevant data.
