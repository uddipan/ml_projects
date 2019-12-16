# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


###############################################################################
# Sample taken from:
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
###############################################################################

# Load dataset
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = read_csv(url, names=names)
    return dataset


# print stats
def print_stats(dataset):
    # shape
    print(dataset.shape)
    # head
    print(dataset.head(20))
    # descriptions
    print(dataset.describe())
    # class distribution
    print(dataset.groupby('class').size())


# visualize the data
def visualize(dataset):
    # box and whisker plots
    dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
    pyplot.show()
    # histograms
    dataset.hist()
    pyplot.show()
    # scatter plot matrix
    scatter_matrix(dataset)
    pyplot.show()


# get different models
def get_models():
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))
    return models


def get_results(dataset, models):
    results = []
    names = []
    for name, model in models:
        # Split-out validation dataset
        array = dataset.values
        X = array[:, 0:4]
        Y = array[:, 4]
        X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=1)
        # Test options and evaluation metric (use k-fold cross validation)
        # Stratified means that each fold or split of the dataset will aim to have the same distribution
        # of example by class as exist in the whole training dataset.
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

    # Make predictions on validation dataset (use SVC)
    model = SVC(gamma='auto')
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    # Evaluate predictions
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))


def main():
    dataset = load_data()
    print_stats(dataset)
    visualize(dataset)
    models = get_models()
    get_results(dataset, models)


if __name__ == "__main__":
    main()
