## Word2Vec

The repo contains code to train word2vec embeddings with some special tokens kept. All tokens in the training corpus in the form "DBPEDIA_ID/\*" will be trained regardless of their frequencies. We use [Gensim](https://github.com/RaRe-Technologies/gensim)'s impmentation with is pythonic and fast on CPU.

### Corpus Information
The corpus is 20.8G, building from [wiki2vec](https://github.com/idio/wiki2vec) containing 11,521,424 entities marked with "DBPEDIA_ID/\*".
The entity lexicons with frequencies are store in corpus/en_entity_lexicons.txt (355MB), sorting by frequency.

### Training details
- embedding dimension: 100
- initial learning rate: 0.025
- minimum learning rate: 1e-4
- window size: 5
- min_count: 5
- entity_min_count: 1
- sample: 0.001(higher frequency words are downsampled in this rate)
- model: skipgram
- negative sampling noise words: 5
- epochs: 5
- sorted words by frequency
- words batch size: 10000

The training process takes ~9.96 hours on s2 with 8 threading workers, 334461 effective words/s,
~7.36 hours on ai with 32 threading workers, 490843 effective words/s.

### Result
The vocabulary included in the pretrained embedding files are 15,902,725 (15M), with a dimension of 100. Total file is ~20G.

### Validation and Evaluation
To validate whether the ENTITIES are kept, you can run `grep "DBPEDIA_ID/*" word2vec.en_entity_text.100d.ai.txt > out.txt`

A simple evaluation is the word relation test "A is to B as C is to D", the accuracy is 62.7% on s2 and 63.2% on ai. For comparison, Google's best word2vec is above 70%.

### References
- Gensim Tutorials [[1]](https://rare-technologies.com/word2vec-tutorial/)[[2]](https://rare-technologies.com/deep-learning-with-word2vec-and-gensim/)[[3]](https://rare-technologies.com/word2vec-in-python-part-two-optimizing/)[[4]](https://rare-technologies.com/parallelizing-word2vec-in-python/)
- [Google Word2Vec](https://code.google.com/archive/p/word2vec/)
