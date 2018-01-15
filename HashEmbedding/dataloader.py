import os
import urllib
import urllib.request
import pickle
import numpy as np

"""
Raw data iterator for the Datasets available from https://arxiv.org/abs/1509.01626
Character-level Convolutional Networks for Text Classification
Xiang Zhang, Junbo Zhao, Yann LeCun
"""

class UniversalArticleDatasetProvider:
    AG_NEWS = 1
    AMAZON_REVIEW_FULL = 2
    AMAZON_REVIEW_POLARITY = 3
    DBPEDIA = 4
    SOGOU_NEWS = 5
    YAHOO_ANSWERS = 6
    YELP_REVIEW_FULL = 7
    YELP_REVIEW_POLARITY = 8

    def __init__(self, dataset, valid_fraction = 0.05):
        """ The datasets are split into train set and test set.
        Pickle train takes the path to the pickled train set file.
        Pickle test takes the path to the pickled test set file.
        """

        pickle_train = None

        if dataset == self.AG_NEWS:
            pickle_train = 'ag_news_csv_train.pkl'
            pickle_test = 'ag_news_csv_test.pkl'
        if dataset == self.AMAZON_REVIEW_FULL:
            pickle_train = 'amazon_review_full_csv_train.pkl'
            pickle_test = 'amazon_review_full_csv_test.pkl'
        if dataset == self.AMAZON_REVIEW_POLARITY:
            pickle_train = 'amazon_review_polarity_csv_train.pkl'
            pickle_test = 'amazon_review_polarity_csv_test.pkl'
        if dataset == self.DBPEDIA:
            pickle_train = 'dbpedia_csv_train.pkl'
            pickle_test = 'dbpedia_csv_test.pkl'
        if dataset == self.SOGOU_NEWS:
            pickle_train = 'sogou_news_csv_train.pkl'
            pickle_test = 'sogou_news_csv_test.pkl'
        if dataset == self.YAHOO_ANSWERS:
            pickle_train = 'yahoo_answers_csv_train.pkl'
            pickle_test = 'yahoo_answers_csv_test.pkl'
        if dataset == self.YELP_REVIEW_FULL:
            pickle_train = 'yelp_review_full_csv_train.pkl'
            pickle_test = 'yelp_review_full_csv_test.pkl'
        if dataset == self.YELP_REVIEW_POLARITY:
            pickle_train = 'yelp_review_polarity_csv_train.pkl'
            pickle_test = 'yelp_review_polarity_csv_test.pkl'

        assert pickle_train is not None
        assert pickle_test is not None

        self.pickle_file_name_train = pickle_train
        self.source_url_train = 'http://www.intellifind.dk/datasets/%s' % self.pickle_file_name_train
        self.pickle_file_name_test = pickle_test
        self.source_url_test = 'http://www.intellifind.dk/datasets/%s' % self.pickle_file_name_test
        self.valid_fraction = valid_fraction

        self.environment = {'config': {'data': {'data_dir': 'data/'}}}

    def raw_train_samples_gen(self):
        """
        There are some changes to dictionaries depending on the dataset used so be careful!
        For exmaple:
        yahoo_answers_csv_train.pkl: {'answer':str, 'class':int, 'question':str, 'title': str}
        yelp_reviews_full_csv_train.pkl: {'class':int, 'text': str}
        yelp_reviews_polarity_csv_train.pkl: {'class':int, 'text': str}
        All others:
        *_train.pkl: {'class':int, 'text':str, 'title':str}
        """

        for single_dict in self.train_samples:
            yield deepcopy(single_dict)

    def raw_valid_samples_gen(self):
        """
        There are some changes to dictionaries depending on the dataset used so be careful!
        For exmaple:
        yahoo_answers_csv_train.pkl: {'answer':str, 'class':int, 'question':str, 'title': str}
        yelp_reviews_full_csv_train.pkl: {'class':int, 'text': str}
        yelp_reviews_polarity_csv_train.pkl: {'class':int, 'text': str}
        All others:
        *_train.pkl: {'class':int, 'text':str, 'title':str}
        """

        for single_dict in self.valid_samples:
            yield deepcopy(single_dict)

    def raw_test_samples_gen(self):
        """
        There are some changes to dictionaries depending on the dataset used so be careful!
        For exmaple:
        yahoo_answers_csv_test.pkl: {'answer':str, 'class':int, 'question':str, 'title': str}
        yelp_reviews_full_csv_test.pkl: {'class':int, 'text': str}
        yelp_reviews_polarity_csv_test.pkl: {'class':int, 'text': str}
        All others:
        *_test.pkl: {'class':int, 'text':str, 'title':str}
        """

        for single_dict in self.test_samples:
            yield deepcopy(single_dict)



    def prepare(self):
        data_dir = self.environment['config']['data']['data_dir']
        corpus_file_train = os.path.join(data_dir, self.pickle_file_name_train)
        corpus_file_test = os.path.join(data_dir, self.pickle_file_name_test)

        if not os.path.exists(data_dir):
            print('creating data directory: %s' % self.environment['config']['data']['data_dir'])
            os.makedirs(data_dir)
        if not os.path.isfile(corpus_file_train):
            print('Downloading training data (~X MB)')
            urllib.request.urlretrieve(self.source_url_train, corpus_file_train)
            print ('Data dowload complete')
        if not os.path.isfile(corpus_file_test):
            print('Downloading test data (~X MB)')
            urllib.request.urlretrieve(self.source_url_test, corpus_file_test)
            print ('Data download complete')

    def load_data(self):
        filename_train = os.path.join(self.environment['config']['data']['data_dir'],
                                      self.pickle_file_name_train)

        filename_test = os.path.join(self.environment['config']['data']['data_dir'],
                                      self.pickle_file_name_test)

        self.train_samples = pickle.load(open(filename_train, 'rb'))
        self.test_samples = pickle.load(open(filename_test, 'rb'))

        count = 0
        for s in self.train_samples:
            s['class_'] = s['class']
            count += 1
        for s in self.test_samples:
            s['class_'] = s['class']

        print('Num samples: %i' %count)

        np.random.seed(1)
        np.random.shuffle(self.train_samples)
        np.random.shuffle(self.test_samples)

        num_samples = len(self.train_samples)
        num_valid_samples = int(num_samples*self.valid_fraction)

        self.valid_samples = self.train_samples[0:num_valid_samples]

        self.train_samples = self.train_samples[num_valid_samples::]

        self.environment['num_validation_samples'] = len(self.valid_samples)
        self.environment['num_test_samples'] = len(self.test_samples)

