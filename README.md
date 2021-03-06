# Rare-Label-One-Hot Encoder

[![Setup Automated](https://img.shields.io/badge/setup-automated-blue?logo=gitpod)](https://gitpod.io/from-referrer/)
![Test passing](https://img.shields.io/badge/Tests-passing-brightgreen.svg)
![Python Version](https://img.shields.io/badge/python-3.6+-brightgreen.svg)
[![PyPI version](https://badge.fury.io/py/RLOHE.svg)](https://badge.fury.io/py/RLOHE)
![Last Commit](https://img.shields.io/github/last-commit/rahulbordoloi/Rare-Label-One-Hot-Enocder?style=flat-square)
[![Open Source Love png2](https://badges.frapsoft.com/os/v2/open-source.png?v=103)](https://github.com/ellerbrock/open-source-badges/)

## About
Wanna One-Hot Encode your Train-Test sets which contains Rare-Labels and also give importance to the top entries? No Worries!

`Rare-Label-One-Hot Encoder` Python Package is there to rescue you out!

It's a Categorical Encoder which can be mostly used with Classical Machine Learning Algorithms in-order to One-Hot-Encode a Feature having huge cardinality and also having rare labels in the Train-Test sets. <br>
Basically, it'll set a threshold (that can be user-defined) of taking up the top categories/entries and treat the rest (least significant) as `others`. It also handles rare label cases in case of mapping the features from Train to Test respectively and vice versa. <br>

You can set the top entries criterion either by `level` which will consider the Top entries according to the threshold set or the other by `amount` which will consider all the entries above the threshold as top entries and rest as `others`.

Rare-Label-One-Hot Encoder is available as `RLOHE` in [PyPI](https://pypi.org/project/RLOHE/).

## Installation

Run the following command on your terminal to install `RLOHE`: 

1 .  Installing the package using `pip`:
```python
pip install RLOHE
```
OR

```python
pip3 install RLOHE
```

2 . Cloning the repository:

```
git clone https://github.com/rahulbordoloi/Rare-Label-One-Hot-Enocder/
cd Rare-Label-One-Hot-Enocder
pip install -e .
```

## Usage

`RLOHE` package contains two functions, namely : <br>

*   __TopLabeledEntries__ : Gives out Top Labeled Entries' Analysis of two given DataFrames.
*   __RareLabelOneHotEncoder__ : Gives out Rare Label One-Hot Encoded DataFrames according to threshold being set and it's criterion of segregation.

It is advised to run `TopLabeledEntries` first in-order to check for the Top Entries and their representation in their respective dataset before going for the encoding as a sanity check.


<h4> Arguments </h4>

1 . For `TopLabeledEntries` Function : <br>

| __Parameters__ | __Description__ |
|    ---         |       ---       |
| __train__ | Refers to the Train Dataset. |
| __test__ | Refers to the Test Dataset. |
| __feature_name__ | Refers to the Feature on which encoding is to be done |
| __threshold__ | Refers to the Top Features Seggregator Limit. |
| __criterion__ | Refers to `level/volume` according to which top entries will be picked up. Check `reference` for more information. |
| __secondary_feature__ | Refers to check amount statistics of another feature with respect to the primary feature. |
| __verbose__ | Refers to variable which controls Output to the console. |
| __return_dataframe__ | Refers to condition for if a dataframe has to be returned or not. |
    

2 . For `RareLabelOneHotEncoder` Function : <br>

| __Parameters__ | __Description__ |
|    ---         |       ---       |
| __train__ | Refers to the Train Dataset. |
| __test__ | Refers to the Test Dataset. |
| __feature_name__ | Refers to the Feature on which encoding is to be done |
| __threshold__ | Refers to the Top Features Seggregator Limit. |
| __criterion__ | Refers to `level/volume` according to which top entries will be picked up. Check `reference` for more information. |
| __verbose__ | Refers to variable which controls Output to the console. |
| __prefix_name__ | Refers to the Prefix Name to be added in front of each new encoded feature. |

 __Reference__ <br>
    *  `level` : Will be considering up top `level` threshold entries for the particular feature, and rest as `BELOW`. <br>
    *  `amount` : Will be considering up the entries above the threshold for the particular feature, and rest as `BELOW`.

Run this script in order to get the Top Entries according to a given threshold!

```python
# Importing Libraries
import RLOHE as encoder
import pandas as pd

# Main Method
if __name__ == '__main__':

    # Reading in Dataset
    train = pd.read_csv('https://raw.githubusercontent.com/rahulbordoloi/Rare-Label-One-Hot-Enocder/main/Data/Train_Data.csv')
    test = pd.read_csv('https://raw.githubusercontent.com/rahulbordoloi/Rare-Label-One-Hot-Enocder/main/Data/Test_Data.csv')
    
    # Displaying out the Top Entries According to the Threshold set.
    encoder.TopLabeledEntries(train, test, feature_name = 'department_info', threshold = 10, secondary_feature = 'cost_to_pay')
```

Run this script in order to get the Rare Label One-Hot Encoded DataFrames according to a given threshold!

```python
# Importing Libraries
import RLOHE as encoder
import pandas as pd

# Main Method
if __name__ == '__main__':

    # Reading in Dataset
    train = pd.read_csv('https://raw.githubusercontent.com/rahulbordoloi/Rare-Label-One-Hot-Enocder/main/Data/Train_Data.csv')
    test = pd.read_csv('https://raw.githubusercontent.com/rahulbordoloi/Rare-Label-One-Hot-Enocder/main/Data/Test_Data.csv')
    
    # Rare Label One-Hot Encoder [Level Wise]
    encodedTrain, encodedTest = encoder.RareLabelOneHotEncoder(train, test, feature_name = 'department_info', threshold = 10,
                                                       criterion = 'level', prefix_name = 'dept')
```

* Checkout Rare Label One-Hot Encoder Implementation in Google Colab : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EDzY3Al5jeZML-V8BkQQXvMFDzWQmctx?usp=sharing)

## Developing `Rare Label One Hot Encoder`

To install `RLOHE`, along with the tools you need to develop and run tests, and execute the following in your virtualenv:

```bash
$ pip install -e .[dev]
```

## Contact Author

Name : Rahul Bordoloi <br>
Website : https://rahulbordoloi.me <br>
Email : rahulbordoloi24@gmail.com <br>

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://gitHub.com/rahulbordoloi/)
