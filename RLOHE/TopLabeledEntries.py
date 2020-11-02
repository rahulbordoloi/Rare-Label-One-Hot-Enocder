# Import Libraries
import pandas as pd


# Top-Labeled Function
def TopLabeledEntries(train = None, test = None, feature_name = None, threshold = 10, criterion = 'level', secondary_feature = None, verbose = True, return_dataframe = False):

    """
    Gives out Top Labeled Entries' Analysis of two given DataFrames.
    :param train: Train Dataset.
    :param test: Test Dataset.
    :param feature_name: Feature on which encoding is to be done.
    :param threshold: Top Features Seggregator Limit.
    :param criterion: `level/volume` according to which top entries will be picked up.
    :param secondary_feature: To check amount statistics of another feature with respect to the primary feature.
    :param verbose: Variable to Control Output to the console.
    :param return_dataframe: condition for if a dataframe has to be returned or not.
    :return: None/Dataframe.

    __Reference__
    *  Level : Will be considering up top `level` threshold entries for the particular feature, and rest as `BELOW`.
    *  Amount : Will be considering up the entries above the threshold for the particular feature, and rest as `BELOW`.
    """

    # Sanity Check [for DataFrames]
    if pd.DataFrame(train).empty or pd.DataFrame(test).empty:
        raise ValueError('Either of the DataFrame is NULL')

    # Creating Copies of the DataFrames
    train_set = train.copy()
    test_set = test.copy()

    # Taking out the Counts of Entries
    try:

        counts_train = train_set.groupby([feature_name])[feature_name].count().sort_values(ascending = False)
        counts_test = test_set.groupby([feature_name])[feature_name].count().sort_values(ascending = False)

    # Stop Execution and Print Error Message if the given feature doesn't exist.
    except:
        raise ValueError(f"Feature '{feature_name}' not found in the DataFrame!")

    # Taking out the Indexes [Names] of the Entries ABOVE the Threshold

    # Check if criterion is `level`
    if criterion == 'level':

        important_entries_train = counts_train[:threshold].index.tolist()
        important_entries_test = counts_test[:threshold].index.tolist()

    # Check if criterion is `volume`
    elif criterion == 'volume':

        important_entries_train = counts_train[counts_train >= threshold].index.tolist()
        important_entries_test = counts_test[counts_test >= threshold].index.tolist()

    # If Criterion Not Found
    else:
        raise ValueError(f"Criterion type '{criterion}' not found!")

    # Taking out the Indexes [Names] of the Entries BELOW the Threshold

    # [Level Criterion]
    if criterion == 'level':

        below_threshold_train = counts_train[threshold:].index
        below_threshold_test = counts_test[threshold:].index

    # [Volume Criterion]
    elif criterion == 'volume':

        below_threshold_train = counts_train[counts_train < threshold].index
        below_threshold_test = counts_test[counts_test < threshold].index

    # Imputing Entries Below Threshold as 'BELOW' to put them into a Single Category
    train_set[feature_name].replace(below_threshold_train, 'BELOW', inplace = True)
    test_set[feature_name].replace(below_threshold_test, 'BELOW', inplace = True)

    # Grouping Entries after Imputing 'BELOW'

    impute_counts_train = train_set[feature_name].value_counts().sort_values(ascending = False)
    impute_counts_test = test_set[feature_name].value_counts().sort_values(ascending = False)

    # Printing the Result if Verbose is True
    if verbose:

        ## Entries Level Statistics
        print()
        print("*" * 16, "Entries Level Statistics", "*" * 16)
        print()
        print(f"Total Unique Entries of '{feature_name}' in Train : {len(train[feature_name].unique().tolist())}")
        print(f"Total Unique Entries of '{feature_name}' in Test : {len(test[feature_name].unique().tolist())}")
        print()

        # Check `criterion` for Output String
        output_String = f"Contribution of Top {threshold} Entries" if criterion == "level" else f"Contribution of Entries <= {threshold}"

        ## Give out Volume Statistics
        print(f"Volume {output_String} [Train] : {round(sum(counts_train[:threshold]) / len(train_set) * 100, 2)} %")
        print(f"Volume {output_String} [Test]  : {round(sum(counts_test[:threshold]) / len(test_set) * 100, 2)} %")
        print()

        # Check for the Presence of Second Feature [Amount Statistics]
        if secondary_feature:
            print(f"Amount {output_String} [Train] : {round(train_set[train_set[feature_name].isin(important_entries_train)][secondary_feature].sum() / sum(train_set[secondary_feature]) * 100, 2)} %")
            print(f"Amount {output_String} [Test]  : {round(test_set[test_set[feature_name].isin(important_entries_test)][secondary_feature].sum() / sum(test_set[secondary_feature]) * 100, 2)} %")

        # Check `criterion` for Output String
        output_String = f"Common Entries in Top {threshold} Entries" if criterion == "level" else f"Common Entries in Entries <= {threshold}"

        ## Data wise Statistics for Important Entries
        print(f"{output_String} of Both the Sets Combined : {len(set(important_entries_train) & set(important_entries_test))}")
        print(f"Entries that are in Train not in Test : {list(set(important_entries_test) - set(important_entries_train))}")
        print(f"Entries that are in Test not in Train : {list(set(important_entries_train) - set(important_entries_test))}")
        print()

        # Check `criterion` for Output String
        output_String = f"Entries Above & Equal to {threshold}" if criterion == "volume" else f"Top {threshold} Entries"

        ## Train-Test wise Final Groupby
        print("*" * 26, f"Train {output_String}", "*" * 26)
        print(impute_counts_train)
        print()
        print("*" * 26, f"Test {output_String}", "*" * 26)
        print(impute_counts_test)
        print()
        print("Note : 'BELOW' Category represents all the entries below the Threshold.")

    # Checking Condition for `return`
    if return_dataframe:
        return train_set, test_set








