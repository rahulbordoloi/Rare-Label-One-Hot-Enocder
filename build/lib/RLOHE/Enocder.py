# Importing Libraries
import pandas as pd
import warnings
from RLOHE.TopLabeledEntries import TopLabeledEntries
from RLOHE.RareLabelOneHotEncoder import RareLabelOneHotEncoder

# Other Settings
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.6f}'.format
pd.set_option('display.max_columns', 256)


# Main Method
if __name__ == '__main__':

    # Loading in Datasets
    train = pd.read_csv('https://raw.githubusercontent.com/rahulbordoloi/Rare-Label-One-Hot-Enocder/main/Data/Train_Data.csv')
    test = pd.read_csv('https://raw.githubusercontent.com/rahulbordoloi/Rare-Label-One-Hot-Enocder/main/Data/Test_Data.csv')

    # Top Labeled Entries in Both of the Sets for Analysis
    TopLabeledEntries(train, test, feature_name = 'department_info', threshold = 10, secondary_feature = 'cost_to_pay')

    '''
    # Rare Label One-Hot Encoder [Level Wise]
    encodedTrain, encodedTest = RareLabelOneHotEncoder(train, test, feature_name = 'department_info', threshold = 10,
                                                       criterion = 'level', prefix_name = 'dept')
    '''

    # Rare Label One-Hot Encoder [Volume Wise]
    encodedTrain, encodedTest = RareLabelOneHotEncoder(train, test, feature_name = 'department_info', threshold = 12500,
                                                       criterion = 'volume', prefix_name = 'dept')

    # Displaying Encoded DataFrame
    print(encodedTrain.head(2))



