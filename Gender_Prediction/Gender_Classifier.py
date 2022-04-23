from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import pandas as pd
import numpy as np
from getpass import getpass
from sklearn.naive_bayes import MultinomialNB


class GenderClassifier:

    def __init__(self, con):
        # Read DataFrame from SQL Query
        self.name_df = pd.read_sql(
            'SELECT * FROM NationalNames', con=con, dtype={'Count': np.int32})

    def group_by_name_gender(self):
        # Create a pandas groupby object by name and gender
        self.namechart = self.name_df.groupby(
            ['Name', 'Gender'], as_index=False)['Count'].sum()
        return self.namechart

    def namechart_pivot(self):
        # Create a pivot table
        self.namechart_diff = self.namechart.reset_index().pivot('Name', 'Gender', 'Count')
        # Replace missing values with 0
        self.namechart_diff = self.namechart_diff.fillna(0)
        # Compute percentage male counts
        self.namechart_diff['Mpercent'] = (
            (self.namechart_diff["M"] - self.namechart_diff["F"])/(self.namechart_diff["M"] + self.namechart_diff["F"]))
        # Define gender based on male percentage being greater than 0.1%
        self.namechart_diff['gender'] = np.where(
            self.namechart_diff['Mpercent'] > 0.001, 'male', 'female')
        return self.namechart_diff

    def name_bigrams(self):
        # Instantiate CountVectorizer from sklearn
        self.char_vectorizer = CountVectorizer(
            analyzer='char', ngram_range=(2, 2))

        # Transform Names into Bigrams
        X = self.char_vectorizer.fit_transform(self.namechart_diff.index)

        # Convert X to a sparse matrix
        X = X.tocsc()

        # Set dependent variable as gender
        y = (self.namechart_diff.gender == 'male').values.astype(np.int)

        # See output of X in the console
        print(X)

        # Put X and y as output
        return X, y

    def train_test_setup(self):
        # Split diff into training and testing sets
        itrain, itest = train_test_split(
            range(self.namechart_diff.shape[0]), train_size=0.7)

        # Create masking
        mask = np.ones(self.namechart_diff.shape[0], dtype='int')

        # Assign train to 1 and test to 0
        mask[itrain], mask[itest] = 1, 0

        # Re-assign mask to locations where mask is 1 returns True
        mask = (mask == 1)

    def train_model(self, X, y):
        # Getting actual bigrams for train set
        Xtrainthis = X[mask]
        Ytrainthis = y[mask]

        # Getting actual bigrams for test set
        Xtestthis = X[~mask]
        Ytestthis = y[~mask]

        # Instantiate Naive Baye's Classifier using Multinomial Distribution
        clf = MultinomialNB(alpha=1)

        # Train the model using .fit()
        clf.fit(Xtrainthis, Ytrainthis)

        # Compute the training and test accuracies
        training_accuracy = clf.score(Xtrainthis, Ytrainthis)
        test_accuracy = clf.score(Xtestthis, Ytestthis)

        # Output accuracies to the console
        print(training_accuracy)
        print(test_accuracy)

        # Return the trained classifier
        return clf

    def predict(self, clf, name):

        # Transform string name into count vectorizer
        name_as_vec = self.char_vectorizer.transform([name])

        # Create a prediction using the trained NB model
        y_pred = clf.predict(name_as_vec)

        # Log prediction based on probability
        if (y_pred == 1):
            print(f"{name} is most likely a male name.")
        else:
            print(f"{name} is most likely a female name.")
