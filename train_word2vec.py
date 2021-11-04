from argparse import ArgumentParser
from typing import List
from pathlib import Path
import gensim
import logging
import os
import re

from multidirectorycorpusreader.multidirectorycorpusreader import MultiDirectoryCorpusReader
from gensim.models.word2vec import Word2Vec
from gensim.utils import simple_tokenize

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH  = os.path.join(SCRIPT_PATH, 'model/')

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

def preprocess_text(text: str) -> List[str]:
    text = text.lower()
    # Remove numbers from text
    text = re.sub('\d+', '', text)
    # Remove newlines
    text = re.sub('\n', '', text)
    # Convert '.' into new lines, so we have one sentence per line
    text = re.sub('\.', '\n', text)
    # Remove all other symbols
    text = re.sub('[!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~\"\']', '', text)
    text = text.replace('\n', ' ')
    sentences = list(simple_tokenize(text=text))
    # Each document should be a list of lists of strings (?!?)
    # sentences = [word_tokenize(s.strip()) for s in text.split('\n') if s != '']
    # sentences = [e for e in sentences if e != []]
    return sentences

def modelname(model_basename: str, model_type: str, vector_size: int, minimum_count: int, epochs: int, window_size: int) -> Path:
    # Just in case someone builds a model with a vector_size < 100
    vs_zero_pad = max(len(str(vector_size)), 3)
    return Path(f'{model_basename}-{model_type}-vs{vector_size:0{vs_zero_pad}}-mc{minimum_count:02}-e{epochs:02}-w{window_size:02}.model')

if __name__ == '__main__':

    usage = """TrainWord2Vec raw text files from multiple directories

    This script will train a Word2Vec model based on director[y|ies] filled with
    plain text files."""

    parser = ArgumentParser(usage=usage)

    parser.add_argument('-d', '--dir',
                        dest='dir_paths',
                        help='Director(y|ies) to read the text files from. Multiple sources are accepted using "dir1,dir2"',
                        required=True,
                        nargs='+',
                        type=str
                        )
    parser.add_argument('-e', '--epochs',
                        default=10,
                        dest='epochs',
                        help='Number of epochs on which to iterate over the text data',
                        type=int
    )
    parser.add_argument('-g', '--glob-filter',
                        default='*.txt',
                        dest='glob_filters',
                        help='Sets a globbing filter(s), e.g. "*.txt" or "*.msg", multiple filters are also accepted using "*.txt,*.msg"',
                        nargs='+',
                        type=str)
    parser.add_argument('-i', '--in-memory',
                        default=False,
                        dest='in_memory',
                        action='store_true',
                        help='Load file content into memory or stream from sources')
    parser.add_argument('-m', '--minimum-count',
                        default=5,
                        dest='minimum_count',
                        help='Determines the minimum number of word instances',
                        type=int)
    parser.add_argument('-o', '--output',
                        dest='model_filename',
                        help='The script writes the model output to this filename, beware that the name is automatically with vector_size, epochs, window_size and minimum_count and ".model"',
                        required=True,
                        type=str)
    parser.add_argument('-p', '--workers',
                        default=8,
                        dest='workers',
                        help='Determines the number of worker to perform parallel processing',
                        type=int)
    parser.add_argument('-r', '--resume-training',
                        dest='resume',
                        help='Allows for continued training on a previous trained model, name the BASE model here',
                        type=str)
    parser.add_argument('-t', '--talky',
                        action='store_true',
                        default=False,
                        dest='verbose',
                        help='Set verbose on for additional output')
    parser.add_argument('-v', '--vector-size',
                        default=100,
                        dest='vector_size',
                        help='Determines the size of the word vectors',
                        type=int)
    parser.add_argument('-w', '--window-size',
                        default=5,
                        dest='window_size',
                        help='Determines the window size',
                        type=int)
    parser.add_argument('-x', '--use-skip-gram',
                        default=False,
                        dest='use_skip_gram',
                        action='store_true',
                        help='Determines whether or not to use skip-gram instead of cbow (not recommended)')

    args = parser.parse_args()

    model_type = 'sg' if args.use_skip_gram else 'cbow'
    model_filename = modelname(
        model_basename=args.model_filename,
        model_type=model_type,
        vector_size=args.vector_size,
        minimum_count=args.minimum_count,
        epochs=args.epochs,
        window_size=args.window_size)

    model_file_path = MODEL_PATH / model_filename
    if model_file_path.is_file():
        raise FileExistsError(f'Aborting, about to overwrite existing file: {str(model_file_path)}')

    # Fiddle with bool args
    in_memory = args.in_memory
    verbose = args.verbose
    use_sg = 1 if args.use_skip_gram else 0

    mdcr = MultiDirectoryCorpusReader(
        input_dirs=args.dir_paths,
        glob_filters=args.glob_filters,
        preprocessor_func=preprocess_text,
        print_progress=verbose,
        in_memory=in_memory)

    if args.resume:
        resume_file = MODEL_PATH / Path(f'{args.resume}.model')
        if not resume_file.is_file():
            raise FileNotFoundError(f'Unable to continue training on model: {str(resume_file)}, file not found')
        model = Word2Vec.load(str(resume_file))
        model.build_vocab(corpus_iterable=mdcr, update=True)
        model.train(corpus_iterable=mdcr, total_examples=len(mdcr), epochs=10)
    else:
        model = gensim.models.Word2Vec(
            sentences=mdcr,
            sg=use_sg,
            vector_size=args.vector_size,
            window=args.window_size,
            min_count=args.minimum_count,
            workers=args.workers)
        logging.info('Training model')
        model.train(mdcr, total_examples=len(mdcr), epochs=10)
        logging.info('Model training completed')

    logging.info(f'Saving model at {str(model_file_path)}')
    model.save(str(model_file_path))
