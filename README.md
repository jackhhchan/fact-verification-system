# Fact Verification System

The Fact Verification System verify claims using from a subset of information from Wikipedia.

The system verify your claim following these 3 components:
  1. Search (uses BM25 algorithm from Elasticsearch)
  2. Sentence Selection (uses NER from SpaCy)
  3. Natural Language Inference (uses a fine-tuned BERT model trained on Tensorflow 2.0)
  
Search returns relevant results from the Wikipedia dataset, then sentences are further filtered from having at least one shared Entity tag. These sentences are fed into a fine-tuned BERT classifier to output either a REFUTE or a SUPPORT label.

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
