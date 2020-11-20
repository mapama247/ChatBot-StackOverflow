import sys, os, re, pickle
import numpy as np
import pandas as pd

from dialogue_manager import DialogueManager
from utils import *

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

sys.path.append("..")

def tfidf_features(X_train, X_test, vectorizer_path, ngrams=(1,1), analyzer='word'):
    """Performs TF-IDF transformation and dumps the model."""

    # Define an objtect that converts a collection of raw documents to a matrix of TF-IDF features
    vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, lowercase=True,
                                 tokenizer=None, stop_words=None, token_pattern='(\S+)',
                                 ngram_range=ngrams, analyzer=analyzer, max_df=1.0, min_df=10)

    X_train = vectorizer.fit_transform(X_train) # Train a vectorizer on X_train data.
    X_test  = vectorizer.transform(X_test)      # Transform X_train and X_test data.

    # Pickle the trained vectorizer to 'vectorizer_path'.
    with open(vectorizer_path, 'wb') as file:
      pickle.dump(vectorizer, file)

    return X_train, X_test

def prepare_data(file_name, size, random_state=0):
    df = pd.read_csv(file_name, sep='\t').sample(size, random_state=random_state)
    df['text'] = df['text'].apply(lambda x: text_prepare(x))
    return df

def intent_classifier(X, y, test_pctg=0.1, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_pctg, random_state=random_state)
    X_train_tfidf, X_test_tfidf = tfidf_features(X_train, X_test, RESOURCE_PATH['TFIDF_VECTORIZER'], ngrams=(1,1), analyzer='word')

    # Train an intent recognizer using LogisticRegression on the train set
    clf = LogisticRegression(C=10, penalty='l2', random_state=0, max_iter=1000)
    clf.fit(X_train_tfidf, y_train)
    pred = clf.predict(X_test_tfidf)
    print('Test accuracy = {}'.format(accuracy_score(y_test,pred))) # around 0.9
    return clf

def language_classifier(X, y, test_pctg=0.2, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_pctg, random_state=random_state)
    vectorizer = pickle.load(open(RESOURCE_PATH['TFIDF_VECTORIZER'], 'rb'))
    X_train_tfidf = vectorizer.transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    # Train one more Logistic Regression classifier for the programming-related questions. It will predict exactly one programming languege.
    clf = OneVsRestClassifier(LogisticRegression(C=5, penalty='l2', random_state=0, max_iter=1000))
    clf.fit(X_train_tfidf, y_train)
    pred = clf.predict(X_test_tfidf)
    print('Test accuracy = {}'.format(accuracy_score(y_test,pred))) # around 0.8
    return clf

def main():
    # The first thing to do is to distinguish programming-related questions from general ones.
    # It would also be good to predict which programming language a particular question refers to in order to speed up question search by a factor of 10.

    # Load 200k examples of each class (dialogue and programming-related) and preprocess the text
    dialogue_df = prepare_data('data/dialogues.tsv', 200000)
    stackoverflow_df = prepare_data('data/tagged_posts.tsv', 200000)

    # Prepare data for binary classification (programming-related or not) on TF-IDF representations of texts.
    X = np.concatenate([dialogue_df['text'].values, stackoverflow_df['title'].values])
    y = ['dialogue'] * dialogue_df.shape[0] + ['stackoverflow'] * stackoverflow_df.shape[0]

    intent_recog = intent_classifier(X,y)

    # Dump the classifier to use it in the running bot
    pickle.dump(intent_recog, open(RESOURCE_PATH['INTENT_RECOGNIZER'], 'wb'))

    # Prepare data for multiclass classification (10 possible languages) on TF-IDF representations of texts.
    X = stackoverflow_df['title'].values
    y = stackoverflow_df['tag'].values

    tagger = language_classifier(X,y)

    # Dump the classifier to use it in the running bot
    pickle.dump(tagger, open(RESOURCE_PATH['TAG_CLASSIFIER'], 'wb'))

    # To find a relevant answer on a question we will use vector representations to calculate similarity between the question and existing threads.
    # Load embeddings that were trained in supervised mode for duplicates detection on the same corpus that is used in search (StackOverflow posts).
    starspace_embeddings, embeddings_dim = load_embeddings('starspace_embeddings.tsv')

    # Load all the entire set of StackOverflow posts (2,171,575 samples)
    posts_df = pd.read_csv('data/tagged_posts.tsv', sep='\t')

    # For each tag, create two data structures that will serve as online search index: tag_post_ids and tag_vectors
    os.makedirs(RESOURCE_PATH['THREAD_EMBEDDINGS_FOLDER'], exist_ok=True)

    for tag, count in counts_by_tag.items():
        tag_posts = posts_df[posts_df['tag']==tag] # filter out all posts about other programming languages
        tag_post_ids = tag_posts['post_id'].values # list of post_ids needed to show the title and link to the thread
        tag_vectors = np.zeros((count, embeddings_dim), dtype=np.float32) # matrix where embeddings for each answer are stored
        for i, title in enumerate(tag_posts['title']):
          tag_vectors[i, :] = question_to_vec(title, starspace_embeddings, embeddings_dim)

        # Dump post ids and vectors to a file.
        filename = os.path.join(RESOURCE_PATH['THREAD_EMBEDDINGS_FOLDER'], os.path.normpath('%s.pkl' % tag))
        pickle.dump((tag_post_ids, tag_vectors), open(filename, 'wb'))


    dialogue_manager = DialogueManager(RESOURCE_PATH)

    questions = ["Hey", "How are you doing?", "What's your hobby?", "How to loop in python?",
                 "How to delete rows in pandas?", "python3 re", "What is AI?",]

    for question in questions:
        answer = dialogue_manager.generate_answer(question)
        print('Q: %s\nA: %s \n' % (question, answer))

if __name__ == "__main__":
    main()
