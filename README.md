# Fact Verification System

The Fact Verification System verify claims using a subset of text data mined from Wikipedia.

The system verify your claim following these 3 components:
  1. Search (uses BM25 algorithm from Elasticsearch)
  2. Sentence Selection (uses NER from SpaCy)
  3. Natural Language Inference (uses a fine-tuned BERT model trained on Tensorflow 2.0)
  
Search returns relevant results from the Wikipedia dataset, then sentences are further filtered from having at least one shared Entity tag. These sentences are fed into a fine-tuned BERT classifier to output either a REFUTE or a SUPPORT label. Each component is managed using a microservices architecture with single client facing API Gateway.

### Key Technologies:
Pytorch 1.4, Tensorflow 2.0, Elasticsearch 7.5.2, PostgreSQL 12, Docker, GCP App Engine, GCP Cloud Run, SpaCy.

### To run:
To run any module or scripts:
```
cd src/
python -m folder.module
```

## src/fact_verification_system/search
connects to Elasticsearch and contains encapsulated queries to return relevant search results.

## src/fact_verification_system/sentence_selection
uses NER from SpaCy to analyse texts, further filtering them before having them classified.

## src/fact_verification_system/classifier
NLI classifier data cleaning (e.g. downsampling), training and inference scripts.

## src/postgres
database storing raw dataset files. Mostly used to construct the dataset for training the natural language inference classifier.

## src/logger
logs any exceptions into a text file. Has async capabilities.
