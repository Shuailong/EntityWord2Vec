import os
import logging
import argparse

from tqdm import tqdm
import sh
import gensim


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def filter_wrapper(entities):
    def filter_ne(word, count, min_count):
        if word in entities:
            return gensim.utils.RULE_KEEP
        else:
            return gensim.utils.RULE_DEFAULT
    return filter_ne


def main(args):
    # Get sentences
    logging.info('Initializing corpus')
    sentences = gensim.models.word2vec.Text8Corpus(args.corpus)
    # Get entities
    logging.info('Read entities')
    lines = int(str(sh.wc('-l', args.entities)).split()[0])
    with open(args.entities, encoding='utf-8') as f:
        entities = set()
        for line in tqdm(f, total=lines):
            entities.add(line.split('\t')[0].strip())
    filter = filter_wrapper(entities)
    # Train models
    logging.info('Start training')
    model = gensim.models.Word2Vec(sentences, size=args.emb_dim, workers=args.workers,
                                   sg=(args.model_type == 'skipgram'), iter=args.num_epochs,
                                   trim_rule=filter, batch_words=args.batch_words)
    # Evaluate
    if args.eval:
        logging.info('Start evaluating')
        model.accuracy(args.eval)

    # Save
    if args.save:
        path = args.save
    else:
        corpus_name = os.path.splitext(os.path.basename(args.corpus))[0]
        path = os.path.join('embeddings', f'word2vec.{corpus_name}.{args.emb_dim}d.txt')
    logging.info('Start saving')
    model.wv.save_word2vec_format(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train word2vec from wiki entity corpus')
    # Files
    parser.add_argument('--corpus', type=str, default='corpus/en_entity_text.txt',
                        required=True)
    parser.add_argument('--entities', type=str, default='corpus/en_entity_lexicons.txt')
    parser.add_argument('--eval', type=str, default='eval/questions-words.txt')
    parser.add_argument('--save', type=str, default=None, help='trained embedding file')

    # Model
    parser.add_argument('--emb-dim', type=int, default=100, help='embedding dimension')
    parser.add_argument('--model-type', type=str, default='skipgram', choices=['skipgram', 'cbow'],
                        help='CBoW or Skip Gram. For large dataset, use skipgram.')

    # Optimizer
    parser.add_argument('--num-epochs', type=int, default=5, help='number of epochs')

    # Runtime
    parser.add_argument('--workers', type=int, default=5, help='workers for multithreading')
    parser.add_argument('--batch-words', type=int, default=10000, help='word batch size')
    args = parser.parse_args()
    main(args)
