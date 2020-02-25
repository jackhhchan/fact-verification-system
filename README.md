# Fact Verification System
Deployed on google cloud at https://fact-verification-system.appspot.com/.

This system runs on a subset of Wikipedia data.
It consist of 3 components. 
  1. Relevant results are retrieved from an Elasticsearch engine.
  2. Sentences are further filtered (such as using NER for Tag matching).
  3. Filtered sentences are passed to a fine-tuned BERT neural network for classification.

### To run:
To run any module or scripts:
e.g.
```
cd src/
python -m fact_verification_system.classifier.scripts.train
```
## Production
### src/production/
Contains the code for api-gateway (sentence selection & classifier) and search deployed on google cloud.

## Development
### src/fact_verification_system/
This contains the search, sentence selection and classifier components.
### src/postgres
Contains code used to access Postgresql database storing the raw string dataset.
### src/logger
Contains code used for logging the any errors into text files at src/logs.
