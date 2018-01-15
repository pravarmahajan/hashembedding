import string
from layers import HashEmbedding, ReduceSum
from keras.layers import Input, Dense, Activation, Embedding
from keras.models import Model
import hashlib
import nltk
import keras
import numpy as np
from keras.callbacks import EarlyStopping

import dataloader
import random

def get_model(embedding, num_classes):
    input_words = Input([None], dtype='int32', name='input_words')

    x = embedding(input_words)
    x = ReduceSum()([x, input_words])

    #x = Dense(50, activation='relu')(x)

    #x = Dense(num_classes)(x)
    x = Activation('softmax')(x)
    model = Model(input=input_words, output=x)
    return model

def word_encoder(w, max_idx):
    # v = hash(w) #
    v = int(hashlib.sha1(w.encode('utf-8')).hexdigest(), 16)
    return (v % (max_idx-1)) + 1


def remove_punct(in_string):
    return ''.join([ch.lower() if ch not in string.punctuation else ' ' for ch in in_string])


def bigram_vectorizer(documents, bigram_dict={}):
    if len(bigram_dict)==0:
        docs2id = [None]*len(documents)
        for (i, document) in enumerate(documents):
            tokens = document.split(' ')
            docs2id[i] = [None]*(len(tokens)-1)
            for j in range(len(tokens)-1):
                key = tokens[j]+"_"+tokens[j+1]
                if key not in bigram_dict:
                    bigram_dict[key] = len(bigram_dict)+1
                docs2id[i][j] = bigram_dict[key]
        return bigram_dict, docs2id
    else:
        docs2id = [None]*len(documents)
        for (i, document) in enumerate(documents):
            tokens = document.split(' ')
            docs2id[i] = [None]*(len(tokens)-1)
            for j in range(len(tokens)-1):
                key = tokens[j]+"_"+tokens[j+1]
                if key not in bigram_dict:
                    docs2id[i][j] = 0
                else:
                    docs2id[i][j] = bigram_dict[key]
        return docs2id


# In[4]:


def input_dropout(docs_as_ids, min_len=4, max_len=100):
    dropped_input = [None]*len(docs_as_ids)
    for i, doc in enumerate(docs_as_ids):
        random_len = random.randrange(min_len, max_len+1)
        idx = max(len(doc)-random_len, 0)
        dropped_input[i] = doc[idx:idx+random_len]
    return dropped_input

def create_dataset():
    dl_obj = dataloader.UniversalArticleDatasetProvider(1, valid_fraction=0.05)
    dl_obj.load_data()

    train_documents = [remove_punct(sample['title'] + " " + sample['text']) for sample in dl_obj.train_samples]
    train_targets = [sample['class'] - 1 for sample in dl_obj.train_samples]

    val_documents = [remove_punct(sample['title'] + " " + sample['text']) for sample in dl_obj.valid_samples]
    val_targets = [sample['class'] - 1 for sample in dl_obj.valid_samples]

    bigram_dict, train_docs2id = bigram_vectorizer(train_documents)
    val_docs2id = bigram_vectorizer(val_documents, bigram_dict)

    train_docs2id = input_dropout(train_docs2id)
    train_docs2id = [d+[0]*(max_len-len(d)) for d in train_docs2id]
    val_docs2id = [d+[0]*(max_len-len(d)) for d in val_docs2id]
    #val_docs2id = input_dropout(val_docs2id)

    #train_docs2id = train_docs2id % max_words
    #val_docs2id = val_docs2id % max_words
    
    return train_docs2id, train_targets, val_docs2id, val_targets

use_hash_embeddings = True
embedding_size = 20
num_buckets = 10**6 # number of buckets in second hashing layer (hash embedding)
max_words = 10**7  # number of buckets in first hashing layer
max_epochs = 50
num_hash_functions = 2
max_len = 150
num_classes = 4

if __name__ == '__main__':

    if use_hash_embeddings:
        embedding = HashEmbedding(max_words, num_buckets, embedding_size, num_hash_functions=num_hash_functions)
    else:
        embedding = Embedding(max_words, embedding_size)

    train_data, train_targets, test_data, test_targets = create_dataset()

    model = get_model(embedding, num_classes)
    metrics = ['accuracy']
    loss = 'sparse_categorical_crossentropy'
    model.compile(optimizer=keras.optimizers.Adam(),loss=loss, metrics=['accuracy'])

    print('Num parameters in model: %i' % model.count_params())
    #model.fit(train_data, train_targets, nb_epoch=max_epochs, validation_data = (test_data, test_targets),
    model.fit(train_data, train_targets, nb_epoch=max_epochs, validation_split = 0.1,
              callbacks=[EarlyStopping(patience=5)])

    #test_result = model.test_on_batch(data_encoded[idx_test::, :], targets[idx_test::])
    #for i, (name, res) in enumerate(zip(model.metrics_names, test_result)):
        #print('%s: %1.4f' % (name, res))
