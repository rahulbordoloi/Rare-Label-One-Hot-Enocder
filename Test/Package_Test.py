# Import Package
import pandas as pd
import RLOHE as r             # Rare Label One-Hot Encoder
import warnings

# Other Settings
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.6f}'.format
pd.set_option('display.max_columns', 256)


# Main Method
if __name__ == '__main__':

    # Loading in Dataset
    train = pd.read_csv('https://raw.githubusercontent.com/rahulbordoloi/Rare-Label-One-Hot-Enocder/main/Data/Train_Data.csv')
    test = pd.read_csv('https://raw.githubusercontent.com/rahulbordoloi/Rare-Label-One-Hot-Enocder/main/Data/Test_Data.csv')

    # Top Labeled Entries in Both of the Sets for Analysis
    r.TopLabeledEntries(train, test, feature_name = 'department_info', threshold = 10, secondary_feature = 'cost_to_pay')

    '''
    # Rare Label One-Hot Encoder [Level Wise]
    encodedTrain, encodedTest = r.RareLabelOneHotEncoder(train, test, feature_name = 'department_info', threshold = 10,
                                                         criterion = 'level', prefix_name = 'dept', verbose = True)
    '''

    # Rare Label One-Hot Encoder [Volume Wise]
    encodedTrain, encodedTest = r.RareLabelOneHotEncoder(train, test, feature_name = 'department_info', threshold = 12500,
                                                         criterion = 'volume', prefix_name = 'dept')

    # Printing out DataFrame [Train]
    print(encodedTrain.head(1))

    # Printing out DataFrame [Test]
    print(encodedTest.head(1))





