from RoBERTa_functions import *

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

df = read_csv()

X_new, y_new = get_XY_data(df)
categories = get_categories(df)
n_categories = get_categories_count(categories)

tokenizer = get_tokenizer()

X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.3, random_state=777) # random_state to reproduce results

X_train = roberta_encode(X_train, tokenizer)
X_test = roberta_encode(X_test, tokenizer)

y_train = np.asarray(y_train, dtype='int32')
y_test = np.asarray(y_test, dtype='int32')

def get_model():
    model_unfit = build_model(n_categories)
    #print_model_summary(model_unfit)
    model = fit_model(model_unfit, X_train, y_train, X_test, y_test)
    return model



 





  