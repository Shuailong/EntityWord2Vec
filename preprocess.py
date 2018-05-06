import os
import argparse
import logging
from collections import Counter

from tqdm import tqdm
from gensim import utils

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def main(args):
    logging.info('preprocessing started.')
    rest = b''
    pbar = tqdm(desc='Reading corpus',
                total=os.path.getsize(args.corpus),
                unit='B', unit_scale=True, unit_divisor=1024)
    if args.uncased:
        args.entities = os.path.splitext(args.entities)[0] + '_uncased.txt'
    with utils.smart_open(args.corpus) as fin,\
            open(args.entities, 'w', encoding='utf-8') as fout:
        lexicons = Counter()
        while True:
            text = rest + fin.read(8192)  # avoid loading the entire file (=1 line) into RAM
            pbar.update(n=8192)
            pbar.set_postfix(text=text[:10])
            if text == rest:  # EOF
                words = utils.to_unicode(text).split()
                for w in words:
                    if w.startswith('DBPEDIA_ID/'):
                        if args.uncased:
                            w = w.lower() 
                        lexicons[w] += 1
                break
            last_token = text.rfind(b' ')  # last token may have been split in two... keep for next iteration
            words, rest = (utils.to_unicode(text[:last_token]).split(),
                           text[last_token:].strip()) if last_token >= 0 else ([], text)
            for w in words:
                if w.startswith('DBPEDIA_ID/'):
                    if args.uncased:
                        w = w.lower() 
                    lexicons[w] += 1
        for lexicon, freq in lexicons.most_common():
            fout.write(f'{lexicon}\t{freq}\n')

    logging.info(f'{len(lexicons)} entities identified and saved to {args.entities}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='corpus preprocessing')
    # Files
    parser.add_argument('--corpus', type=str, default='corpus/en_entity_text.txt',
                        required=True)
    parser.add_argument('--entities', type=str, default='corpus/en_entity_lexicons.txt',
                        help='entity names starts with DBPEDIA_ID/')
    parser.add_argument('--uncased', action='store_true', help='lower case')
    args = parser.parse_args()
    main(args)

