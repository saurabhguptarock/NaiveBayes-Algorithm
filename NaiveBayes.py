import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('mushrooms.csv')

le = LabelEncoder()
ds = df.apply(le.fit_transform)

data = ds.values
X = data[:, 1:]
Y = data[:, 0]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


def prior_prob(y_train, label):
    total_examples = y_train.shape[0]
    class_examples = np.sum(y_train == label)

    return (class_examples) / float(total_examples)


def cond_prob(x_train, y_train, feature_col, feature_val, label):
    x_filter = x_train[y_train == label]
    numerator = np.sum(x_filter[:, feature_col] == feature_val)
    denomenator = np.sum(y_train == label)

    return numerator / float(denomenator)


def predict(x_train, y_train, xtest):
    classes = np.unique(y_train)
    post_probs = []
    n_features = x_train.shape[1]

    for label in classes:
        likelihood = 1.0
        for f in range(n_features):
            cond = cond_prob(x_train, y_train, f, xtest[f], label)
            likelihood *= cond
        prior = prior_prob(y_train, label)
        post = likelihood * prior
        post_probs.append(post)
    pred = np.argmax(post_probs)
    return pred


def score(x_train, y_train, x_test, y_test):
    pred = []
    for i in range(x_test.shape[0]):
        pred_label = predict(x_train, y_train, x_test[i])
        pred.append(pred_label)
    pred = np.array(pred)
    accuracy = np.sum(pred == y_test) / y_test.shape[0]
    return accuracy


print(score(x_train, y_train, x_test, y_test))
