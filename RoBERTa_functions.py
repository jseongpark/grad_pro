from RoBERTa_utils import *
import pandas as pd
import numpy as np

import os
import tensorflow as tf

from tokenizers import BertWordPieceTokenizer
from transformers import TFRobertaModel
from sklearn.metrics import f1_score

def get_current_path():
    return os.getcwd()

def read_csv():
    df = pd.read_csv(get_current_path() + file_path)
    df.head()
    df.count()
    return df

def get_XY_data(df):
    X_data = df[['text']].to_numpy().reshape(-1)
    y_data = df[['category']].to_numpy().reshape(-1)
    return X_data, y_data

def get_categories(df):
    categories = df[['category']].values.reshape(-1)
    categories = df['category'].unique()
    return categories

def print_text_counts(X_data):
    n_texts = len(X_data)
    print('Texts in dataset: %d' % n_texts)

def get_categories_count(categories):
    n_categories = len(categories)
    return n_categories
    #print('Number of categories: %d' % n_categories)

def get_categiry_to_name(y_data):
    category_to_id = {}
    category_to_name = {}

    for index, c in enumerate(y_data):
        if c in category_to_id:
            category_id = category_to_id[c]
        else:
            category_id = len(category_to_id)
            category_to_id[c] = category_id
            category_to_name[category_id] = c
        
        y_data[index] = category_id
    return category_to_name

def roberta_encode(texts, tokenizer):
    ct = len(texts)
    input_ids = np.ones((ct, MAX_LEN), dtype='int32')
    attention_mask = np.zeros((ct, MAX_LEN), dtype='int32')
    token_type_ids = np.zeros((ct, MAX_LEN), dtype='int32') # Not used in text classification

    for k, text in enumerate(texts):
        # Tokenize
        tok_text = tokenizer.encode(text)
        #encode_test(tokenizer, text)

        # Truncate and convert tokens to numerical IDs
        enc_text = tok_text.ids[:(MAX_LEN-2)]
        
        input_length = len(enc_text) + 2
        input_length = input_length if input_length < MAX_LEN else MAX_LEN
        
        # Add tokens [CLS] and [SEP] at the beginning and the end
        input_ids[k,:input_length] = np.asarray([0] + enc_text + [2], dtype='int32')
        
        # Set to 1s in the attention input
        attention_mask[k,:input_length] = 1

    return {
        'input_word_ids': input_ids,
        'input_mask': attention_mask,
        'input_type_ids': token_type_ids
    }

def encode_test(tokenizer, input_str):
    output = tokenizer.encode(input_str)
    print('=>idx   : %s'%output.ids)
    print('=>tokens: %s'%output.tokens)
    print('=>offset: %s'%output.offsets)
    print('=>decode: %s\n'%tokenizer.decode(output.ids))

def train_tokenizer():
    data_file = get_current_path() + mecab_vocab_path
    vocab_size = 30000
    limit_alphabet = 6000
    min_frequency = 5

    tokenizer = BertWordPieceTokenizer(lowercase=False)
    tokenizer.train(files=data_file,
                    vocab_size=vocab_size,
                    limit_alphabet=limit_alphabet,
                    min_frequency=min_frequency)
    tokenizer.save_model(get_current_path() + '/Data/tokenizer_model')
    return tokenizer

def get_tokenizer():
    vocab_path = get_current_path() + pretrained_tokenizer_path
    
    tokenizer = BertWordPieceTokenizer(
        vocab_path, 
        clean_text=False,
        handle_chinese_chars=True,
        strip_accents=False,
        lowercase=False,
    )
    return tokenizer

def build_model(n_categories):
    #with strategy.scope():
    input_word_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_word_ids')
    input_mask = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_mask')
    input_type_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_type_ids')

    roberta_model = TFRobertaModel.from_pretrained(MODEL_NAME)
    x = roberta_model(input_word_ids, attention_mask=input_mask, token_type_ids=input_type_ids)

    x = x[0]

    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(n_categories, activation='softmax')(x)

    model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=x)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    return model

def print_model_summary(model):
    #with strategy.scope():
    model.summary()

def fit_model(model, X_train, y_train, X_test, y_test):
    #with strategy.scope():
    print('Training...')
    history = model.fit(X_train,
                        y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        verbose=1,
                        validation_data=(X_test, y_test))
    return model

def print_model_accuracy(model, X_test, y_test):
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))  

def print_model_f1score(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = [np.argmax(i) for i in model.predict(X_test)]
    f1= f1_score(y_test, y_pred)
    print('f1 score: ',f1)