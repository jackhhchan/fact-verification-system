# Fact Verification System

This is a mini project inspired by the Natural Language Processing subject in Unimelb. It follows a funnel approach where sentences relevant to the input claim are returned then filtered and finally classified for whether it supports or refutes your claim. 

You may access the the demo [here](http://www.fact-verification-system.appspot.com/).

***

The system verifies your claim in 3 simple steps:

1. Your input claim is used to return relevant documents from a search engine holding Wikipedia pages.
2. The returned documents are further narrowed down using Entity Linking.
3. Infer for each filtered sentence whether it supports or refutes the claim.

The components that are these 3 steps can simply be known as the Search Engine, Entity Linking and Natural Language Inference respectively.

***

**Architecture:**

![image-20201004182358914](/home/jack/.config/Typora/typora-user-images/image-20201004182358914.png)

<u>Key Technologies Used:</u>

Pytorch, React, Flask and GCP (App Engine, Compute Engine, Cloud Run)

***

#### 1. Search Engine

The search engine uses the bm25 ranking algorithm to return the relevant documents for step 2.

###### Okapi BM25 Algorithm

The [Okapi BM25 algorithm](https://en.wikipedia.org/wiki/Okapi_BM25#:~:text=BM25%20is%20a%20bag%2Dof,slightly%20different%20components%20and%20parameters.) scores each document in the inverted index database based on the classic idea of [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf). The implementation used in this project is sourced [here](https://github.com/dorianbrown/rank_bm25).

###### Available Documents

A subset of Wikipedia (~ 250,000 pages) is randomly sampled to be stored in an in-memory database. This is only done to save cost of hosting it on GCP.

#### 2. Entity Linking

[Entity linking](https://en.wikipedia.org/wiki/Entity_linking) is the idea of mapping words of interest in a sentence (e.g. a person's name or locations) to a target knowledge base, in this case our input claim. At least one word of interest must match between the returned sentence and your input claim to progress to step 3. Words of interest are extracted using NER tagger from [spacy]().

#### 3. Natural Language Inference

A binary classifier is trained on ~25 million sentences from Wikipedia. Sentences are first converted into BERT word embedding. [BERT](http://jalammar.github.io/illustrated-bert/) is a language model trained on the Wikipedia dataset by Google using the [Transformer](https://jalammar.github.io/illustrated-transformer/) architecture. This project's implementation extends the pytorch module from [BERT from huggingface](https://huggingface.co/transformers/model_doc/bert.html?highlight=bert).

The extension is as follows:

```
class BERTNli(BertModel):
    """ Fine-Tuned BERT model for natural language inference."""
    
    def __init__(self, config):
        super().__init__(config)
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        
    
    def forward(self,
               input_ids=None,
               token_type_ids=None):
        x = super(BERTNli, self)\
        		.forward(input_ids=input_ids, token_type_ids=token_type_ids)
        (_, pooled) = x    # see huggingface's doc.
        x = F.relu(self.fc1(pooled))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
```





###### Dependencies

- rank-bm25
- spacy
- transformers (huggingface)

