from RoBERTa import *
from collections import Counter

import math
import statistics
import regex as re
import seaborn as sns
import matplotlib.pyplot as plt

def show_counter_category(categories):
    counter_categories = Counter(categories)
    category_names = counter_categories.keys()
    category_values = counter_categories.values()

    y_pos = np.arange(len(category_names))

    plt.figure(1, figsize=(10, 5))
    plt.bar(y_pos, category_values, align='center', alpha=0.5)
    plt.xticks(y_pos, category_names)
    plt.ylabel('Number of texts')
    plt.title('Distribution of texts per category')
    plt.gca().yaxis.grid(True)
    plt.show()

    print(counter_categories)

def display_lengths_histograms(df_stats, n_cols=3):
    categories = df['category'].unique()
    n_rows = math.ceil(len(categories) / n_cols)
    
    plt.figure(figsize=(15, 8))
    plt.suptitle('Distribution of lengths')
    
    # Subplot of all lengths
    plt.subplot(n_rows, n_cols, 1)
    plt.title('All categories')
    lengths = df_stats['global']['lengths']
    plt.hist(lengths, color='r')

    # Subplot of each category
    index_subplot = 2
    for c in categories:
        plt.subplot(n_rows, n_cols, index_subplot)
        plt.title('Category: %s' % c)
        
        lengths = df_stats['per_category']['lengths'][c]
        plt.hist(lengths, color='b')

        index_subplot += 1

    plt.show()

def calculate_stats(df, split_char=' '):
    categories = df['category'].unique()
    
    all_lengths = []
    per_category = {
        'lengths': {c:[] for c in categories},
        'mean': {c:0 for c in categories},
        'stdev': {c:0 for c in categories}
    }

    for index, row in df.iterrows():
        text = row['text']
        text = re.sub(r"\s+", ' ', text) # Normalize
        text = text.split(split_char)
        l = len(text)
        
        category = row['category']
        
        all_lengths.append(l)
        per_category['lengths'][category].append(l)
    
    for c in categories:
        per_category['mean'][c] = statistics.mean(per_category['lengths'][c])
        per_category['stdev'][c] = statistics.stdev(per_category['lengths'][c])
    
    global_stats = {
        'mean': statistics.mean(all_lengths),
        'stdev': statistics.stdev(all_lengths),
        'lengths': all_lengths
    }
    
    return {
        'global': global_stats,
        'per_category': pd.DataFrame(per_category)
    }

def show_df_stats(df): 
    df_stats = calculate_stats(df)
    df_stats['per_category']

def show_accuracy(history):
    plt.figure(figsize=(10, 10))
    plt.title('Accuracy')

    xaxis = np.arange(len(history.history['accuracy']))
    plt.plot(xaxis, history.history['accuracy'], label='Train set')
    plt.plot(xaxis, history.history['val_accuracy'], label='Validation set')
    plt.legend()

def show_plot_confusion_matrix(X_test, y_test, model):
    y_pred = model.predict(X_test)
    y_pred = [np.argmax(i) for i in model.predict(X_test)]

    con_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_pred).numpy()

    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    label_names = list(range(len(con_mat_norm)))

    con_mat_df = pd.DataFrame(con_mat_norm,
                              index=label_names, 
                              columns=label_names)

    figure = plt.figure(figsize=(10, 10))
    sns.heatmap(con_mat_df, cmap=plt.cm.Blues, annot=True)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')