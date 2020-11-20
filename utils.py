import re, nltk, pickle
import numpy as np

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'data/word_embeddings.tsv',
}

def text_prepare(text):
    """Performs tokenization and simple preprocessing."""

    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    good_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = good_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(path):
    """Loads pre-trained word embeddings from tsv file."""

    embeds = dict() # dictionary mapping words to vectors
    for line in open(path, encoding='utf-8'):
        row = line.strip().split('\t')
        embeds[row[0]] = np.array(row[1:], dtype=np.float32)

    embeddings_dim = embeds[list(embeds)[0]].shape[0]

    return embeds, embeddings_dim


def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""

    words = question.split()

    counter = 0
    res = np.zeros(dim)
    for word in words:
        if word in embeddings:
            res += np.array(embeddings[word])
            counter += 1
    if counter!=0:
        return res/counter # mean of all word embeddings
    else:
        return res # vector of zeros
        

def unpickle_file(filename):
    """Returns the result of unpickling the file content."""

    with open(filename, 'rb') as f:
        return pickle.load(f)
