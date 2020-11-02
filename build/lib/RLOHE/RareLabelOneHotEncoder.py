# Import Libraries & Files
import pandas as pd
from RLOHE.TopLabeledEntries import TopLabeledEntries


# Rare-Label-One-Hot Encoder
def RareLabelOneHotEncoder(train = None, test = None, feature_name = None, threshold = 10, criterion = 'level',
                           verbose = False, prefix_name = None):

    """
    Gives out Rare Label One-Hot Encoded DataFrames according to threshold being set and it's criterion of segregation.
    :param train: Train Dataset.
    :param test: Test Dataset.
    :param feature_name: Feature on which encoding is to be done.
    :param threshold: Top Features Segregation Limit.
    :param criterion: `level/volume` according to which top entries will be picked up.
    :param verbose: Variable to Control Output to the console.
    :param prefix_name: Prefix Name to be added in front of each new encoded feature.
    :return: None/DataFrame.

    __Reference__
    *  Level : Will be considering up top `level` threshold entries for the particular feature, and rest as `BELOW`.
    *  Amount : Will be considering up the entries above the threshold for the particular feature, and rest as `BELOW`.
    """

    # Creating a Copy()
    train_set, test_set = TopLabeledEntries(train, test, feature_name, threshold, criterion, verbose = False,
                                            return_dataframe = True)

    # One-Hot Encoding the Top [Threshold] Features
    train_set = pd.get_dummies(train_set, columns = [feature_name], prefix = prefix_name)
    test_set = pd.get_dummies(test_set, columns = [feature_name], prefix = prefix_name)

    # Taking out the Encoded Features
    train_encoded_cols = train_set.filter(regex = f"{prefix_name}.*").columns
    test_encoded_cols = test_set.filter(regex = f"{prefix_name}.*").columns

    # Checking out Missing Feature(s)
    train_set_missing_cols = list(set(test_encoded_cols) - set(train_encoded_cols))
    test_set_missing_cols = list(set(train_encoded_cols) - set(test_encoded_cols))

    # Printing the Result if Verbose is True
    if verbose:
        print()
        print(f"Missing Features in Train : {train_set_missing_cols}")
        print(f"Missing Features in Test : {test_set_missing_cols}")
        print()

    # Adding up Missing Feature(s) in the Respective DataFrames [0-Padding]

    for i in range(len(train_set_missing_cols)):
        train_set[train_set_missing_cols[i]] = [0] * len(train)

    for i in range(len(test_set_missing_cols)):
        test_set[test_set_missing_cols[i]] = [0] * len(test)

    # Printing the Result if Verbose is True
    if verbose:

        print("*" * 15, "New Train Head", "*" * 15)
        print(train_set.head(1))
        print("*" * 15, "New Test Head", "*" * 15)
        print(test_set.head(1))

    # Printing out Operation Done
    print(f'`{feature_name}` Rare Label One-Hot Encoded!')

    # Returning the DataFrame
    return train_set, test_set




