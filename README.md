# Deep-Learning-Company-Bankruptcy-Prediction

The objective of the project is to perform the classification of companies bankrupt using Deep Learning. The dataset is taken from [kaggle](https://www.kaggle.com/fedesoriano/company-bankruptcy-prediction) and it contains the data of some companies with a column that indicates if the company is bankrupt or not.

This readme will explain the dataset structure, how the project works and the best network achieved so far.

# Index

- [Project structure](#project-structure)
- [Dataset structure](#dataset-structure)
- [How the project works](#how-the-project-works)
  * [Dataset download and import](#dataset-download-and-import)
  * [Dataset analysis](#dataset-analysis)
    + [Shape](#shape)
    + [Null Values](#null-values)
    + [Balance](#balance)
    + [Check Values](#check-values)
  * [Data Normalization](#data-normalization)
  * [Balance Dataset](#balance-dataset)
  * [Split data into training, validation and test set](#split-data-into-training--validation-and-test-set)
  * [Network creation](#network-creation)
    + [Network Evaluate](#network-evaluate)
    + [Save Model](#save-model)
- [Best model analysis](#best-model-analysis)
  * [Network structure](#network-structure)
  * [Model Loss](#model-loss)
  * [Model Accuracy](#model-accuracy)
  * [Test set performance with SMOTE](#test-set-performance-with-smote)
    + [Confusion Matrix](#confusion-matrix)
  * [Test set performance without SMOTE](#test-set-performance-without-smote)
    + [Confusion Matrix](#confusion-matrix-1)
- [How to run the project](#how-to-run-the-project)
  * [Load the model](#load-the-model)

# Project structure
```
.
|
| Folders
├── data   # Folder with the dataset
│   └── data.csv
├── network   # Folder with the network
│   └── network.zip
├── img   # Images of the graphs for the analysis report
|
| Script
├── analyze_dataset.py   # Function to analyze dataset
├── deep_learing.py   # Function to perform deep learning
├── main.py
└── variables.py   # File with the variables to configure the code
```

# Dataset structure

The data were collected from the Taiwan Economic Journal for the years 1999 to 2009. Company bankruptcy was defined based on the business regulations of the Taiwan Stock Exchange.

The columns of the dataset are the following:

* Y - Bankrupt?: Class label
* X1 - ROA(C) before interest and depreciation before interest: Return On Total Assets(C)
* X2 - ROA(A) before interest and % after tax: Return On Total Assets(A)
* X3 - ROA(B) before interest and depreciation after tax: Return On Total Assets(B)
* X4 - Operating Gross Margin: Gross Profit/Net Sales
* X5 - Realized Sales Gross Margin: Realized Gross Profit/Net Sales
* X6 - Operating Profit Rate: Operating Income/Net Sales
* X7 - Pre-tax net Interest Rate: Pre-Tax Income/Net Sales
* X8 - After-tax net Interest Rate: Net Income/Net Sales
* X9 - Non-industry income and expenditure/revenue: Net Non-operating Income Ratio
* X10 - Continuous interest rate (after tax): Net Income-Exclude Disposal Gain or Loss/Net Sales
* X11 - Operating Expense Rate: Operating Expenses/Net Sales
* X12 - Research and development expense rate: (Research and Development Expenses)/Net Sales
* X13 - Cash flow rate: Cash Flow from Operating/Current Liabilities
* X14 - Interest-bearing debt interest rate: Interest-bearing Debt/Equity
* X15 - Tax rate (A): Effective Tax Rate
* X16 - Net Value Per Share (B): Book Value Per Share(B)
* X17 - Net Value Per Share (A): Book Value Per Share(A)
* X18 - Net Value Per Share (C): Book Value Per Share(C)
* X19 - Persistent EPS in the Last Four Seasons: EPS-Net Income
* X20 - Cash Flow Per Share
* X21 - Revenue Per Share (Yuan ¥): Sales Per Share
* X22 - Operating Profit Per Share (Yuan ¥): Operating Income Per Share
* X23 - Per Share Net profit before tax (Yuan ¥): Pretax Income Per Share
* X24 - Realized Sales Gross Profit Growth Rate
* X25 - Operating Profit Growth Rate: Operating Income Growth
* X26 - After-tax Net Profit Growth Rate: Net Income Growth
* X27 - Regular Net Profit Growth Rate: Continuing Operating Income after Tax Growth
* X28 - Continuous Net Profit Growth Rate: Net Income-Excluding Disposal Gain or Loss Growth
* X29 - Total Asset Growth Rate: Total Asset Growth
* X30 - Net Value Growth Rate: Total Equity Growth
* X31 - Total Asset Return Growth Rate Ratio: Return on Total Asset Growth
* X32 - Cash Reinvestment %: Cash Reinvestment Ratio
* X33 - Current Ratio
* X34 - Quick Ratio: Acid Test
* X35 - Interest Expense Ratio: Interest Expenses/Total Revenue
* X36 - Total debt/Total net worth: Total Liability/Equity Ratio
* X37 - Debt ratio %: Liability/Total Assets
* X38 - Net worth/Assets: Equity/Total Assets
* X39 - Long-term fund suitability ratio (A): (Long-term Liability+Equity)/Fixed Assets
* X40 - Borrowing dependency: Cost of Interest-bearing Debt
* X41 - Contingent liabilities/Net worth: Contingent Liability/Equity
* X42 - Operating profit/Paid-in capital: Operating Income/Capital
* X43 - Net profit before tax/Paid-in capital: Pretax Income/Capital
* X44 - Inventory and accounts receivable/Net value: (Inventory+Accounts Receivables)/Equity
* X45 - Total Asset Turnover
* X46 - Accounts Receivable Turnover
* X47 - Average Collection Days: Days Receivable Outstanding
* X48 - Inventory Turnover Rate (times)
* X49 - Fixed Assets Turnover Frequency
* X50 - Net Worth Turnover Rate (times): Equity Turnover
* X51 - Revenue per person: Sales Per Employee
* X52 - Operating profit per person: Operation Income Per Employee
* X53 - Allocation rate per person: Fixed Assets Per Employee
* X54 - Working Capital to Total Assets
* X55 - Quick Assets/Total Assets
* X56 - Current Assets/Total Assets
* X57 - Cash/Total Assets
* X58 - Quick Assets/Current Liability
* X59 - Cash/Current Liability
* X60 - Current Liability to Assets
* X61 - Operating Funds to Liability
* X62 - Inventory/Working Capital
* X63 - Inventory/Current Liability
* X64 - Current Liabilities/Liability
* X65 - Working Capital/Equity
* X66 - Current Liabilities/Equity
* X67 - Long-term Liability to Current Assets
* X68 - Retained Earnings to Total Assets
* X69 - Total income/Total expense
* X70 - Total expense/Assets
* X71 - Current Asset Turnover Rate: Current Assets to Sales
* X72 - Quick Asset Turnover Rate: Quick Assets to Sales
* X73 - Working capitcal Turnover Rate: Working Capital to Sales
* X74 - Cash Turnover Rate: Cash to Sales
* X75 - Cash Flow to Sales
* X76 - Fixed Assets to Assets
* X77 - Current Liability to Liability
* X78 - Current Liability to Equity
* X79 - Equity to Long-term Liability
* X80 - Cash Flow to Total Assets
* X81 - Cash Flow to Liability
* X82 - CFO to Assets
* X83 - Cash Flow to Equity
* X84 - Current Liability to Current Assets
* X85 - Liability-Assets Flag: 1 if Total Liability exceeds Total Assets, 0 otherwise
* X86 - Net Income to Total Assets
* X87 - Total assets to GNP price
* X88 - No-credit Interval
* X89 - Gross Profit to Sales
* X90 - Net Income to Stockholder's Equity
* X91 - Liability to Equity
* X92 - Degree of Financial Leverage (DFL)
* X93 - Interest Coverage Ratio (Interest expense to EBIT)
* X94 - Net Income Flag: 1 if Net Income is Negative for the last two years, 0 otherwise
* X95 - Equity to Liability

The first column indicates with a 0 no bankrupt and with a 1 the bankrupt.

# How the project works

This section will show how the project works.

## Dataset download and import
In the first part of the code there is the import of the dataset into a pandas dataframe. This is a csv file in the data folder. This is a csv file in the **data** folder.

## Dataset analysis
In the next part there will be an analysis of the dataset.

### Shape
The first thing done is to check the shape of the dataframe: **(6819,96)**

### Null Values
The second thing is to check if there are null values. This dataset has no null values.

### Balance
Next it's important to check the balance of the dataset, because if there is a class that has more rows than the other the classification will have a good accuracy but it won't perform well on the minor class.

| Class | Number | Percentage |
|:-----:|:------:|:----------:|
|   0   |  6599  |   96.77 %  |
|   1   |  220   |   3.23 %   |

![class_balance_bar](img/class_balance_bar.png)

![class_balance_pie](img/class_balance_pie.png)

So the dataset is strongly unbalanced, the bankrupt class is only 3.27%. This means that there will be a step in which using SMOTE the dataset will be balanced.

### Check Values
It is important also to check if all the data are normalized. So if the code finds some values that are bigger than 1 and lower than 0, a normalization step will be performed. 

The dataset isn't all normalized.

### Outliers
We also checked if there are many outliers using a box plot chart.

![Outliers](img/boxplot.png)

As we can see many of the columns have outliers. So in a future step there will be the possibility to fix them.

## Capping and Flooring

The code gives the possibility to fix the outlier problem. If we want to fix them the code will perform the capping and flooring. 

![Capping-Flooring](img/outliers.png)

We computed the 25th percentile and the 75th percentile, obtaining Q1 and Q3. Next we computed the [Interquartile Range (IQR)](https://en.wikipedia.org/wiki/Interquartile_range) and then we computed the “Minimum” and the “Maximum”. 
Basically when we find an outlier that is lower than the “Minimum” we change its value with the “Minimum”. The same thing applied to the “Maximum”.


## Data Normalization
The next step is to normalize all the data. This process will use the ```StandardScaler()```. This scaler uses the mean and the standard deviation to set all values to between 0 and 1.

## Split into training and test set
The next step is to split the dataset into training and test sets. We decided that the training set will be 90% of the dataset.
|Set|Percentage|Rows|
|:-:|:-:|:-:|
|Training|90 %|6137|
|Test|10%|682|

## Balance Dataset
Next we balanced the training set using [**SMOTE**](https://towardsdatascience.com/applying-smote-for-class-imbalance-with-just-a-few-lines-of-code-python-cdf603e58688) (Synthetic Minority Oversampling Technique).

The dataset will be filled with new data and it will be balanced.

The shape of the training set depends on the number of the class 0, so it will be different each time. 

After the balance the training set will look like this:  
![class_balance_pie_post](img/class_balance_pie_post.png)

## Split data into training, validation and test set

After we split the training set into **train** and **validation** sets. So we will have these dataframes:
* ```x_train```: The training set data
* ```y_train```: The training set label
* ```x_valid```: The validation set data
* ```y_valid```: The validation set label
* ```x_test```: The validation set data
* ```y_test```: The validation set label

The size will be something like:

|   | Training | Validation | Test |
|:-:|:---:|:----------:|:----:|
|Percent of the dataset|72%|18%|10%|

## Network creation
Here there are two options:
1. Create a new model
2. Load a model
With the first option we need to create a model and train it. While with the other we can just import the model and test it.

### Network Evaluate
Next there is the network evaluation. We evaluate the network with the original test set, so we can see how it performs on non balanced data, like the real world. 

The code will plot useful data to understand how well the model is made:
* The model loss graph
* The model accuracy graph
* The performance of the test set using the balanced dataset
* The confusion matrix using the balanced dataset
* The performance of the test set using the original dataset
* The confusion matrix using the original dataset

### Save Model
In the code there is also the possibility to save the created model. In the ```variables.py``` it is possible to choose the name. The model will be in a folder in the network folder.

# Best model analysis
We tested the code many times, but we won’t insert all the networks obtained. We will show just the best network obtained. 

The best method found doesn't use the flooring and capping, so we have outliers in the dataset.

## Network structure

The network used has this structure:

![Network](img/network.png)

For each Dense layer except the last one there is **relu** as activation function. In the last Dense layer there is the **sigmoid** activation function.

In all Dense layers in the middle of the network there is also the **l2 kernel regularizer** setted with (0.001).

The optimizer is **RMSprop** with the learning rate set at 0.001.

The **loss function** is the binary crossentropy.

We trained the network for 200 epochs.

```python
model = keras.models.Sequential()
model.add(keras.layers.Dense(128, activation='relu', input_shape=(95,)))
model.add(keras.layers.Dense(64,kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(32,kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(16,kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
```

## Model Loss
![Model Loss](img/model_loss.png)

As we can see there are some spikes in the Validation but overall it follows the Training loss. So there is no underfitting and no overfitting.

## Model Accuracy

![Model Accuracy](img/model_accuracy.png)

As we can see there are some spikes in the accuracy but overall the Validation accuracy follows the Training accuracy. 

## Test set performance

In this section we will see how well the network performs on the test set.

| Accuracy | Loss |
|:--------:|:----:|
| 95.89 % | 0.3164|

| Class | Precision | Recall | f1-score | support |
|:-----:|:---------:|:------:|:--------:|:-------:|
| 0 | 0.98 | 0.97 | 0.98 | 660 |
| 1 | 0.40 | 0.55 | 0.46 | 22 |
||
|Macro avg|0.69|0.76|0.72|682|
|Weighted avg|0.97|0.96|0.96|682|

As we can see the results are really good. We have a high accuracy on the test set.

### Confusion Matrix

![Conf_Matr](img/conf_matr.png)

The confusion matrix confirms what was said in the previous paragraph. Almost the entirety of the **No Bankrupt** class was classified correctly, while with the **Bankrupt** there were a lot of errors.
So, the model doesn’t perform well in the lower class.

# How to run the project
Here we will explain how to run the code. It’s important to have python installed

1. Clone the repository
```bash
git clone https://github.com/thisispivi/Deep-Learning-Company-Bankruptcy-Prediction.git
```

2. Run these lines on a terminal
```bash
pip install sklearn
pip install imblearn
pip install pandas
pip install matplotlib
pip install tensorflow
pip install seaborn
```

3. Open the ```variables.py``` file and configure the variables:
* **train_model** -> True: the network will be trained / False: network won't' be trained
* **model_loss** -> True: plot the model loss / False: don't plot the model loss
* **model_accuracy** -> True: plot the model accuracy / False: don't plot the model accuracy
* **evaluate_model** -> True: evaluate the model / False: don't evaluate the model
* **conf_matr** -> True: plot the confusion matrix / False: don't plot the confusion matrix
vplot_model** -> True: plot the structure of the network / False: don't plot the structure of the network
* **save_model** -> True: save the model / False: don't save the model
* **load_model** -> True: load a model in the network folder / False: don't load the model
* **save_figure** -> True: save the image of the plots / False: show the plots
* **substitute** -> True: substitute outliers / False: don't fix outliers 

4. Run the code:
```bash
python main.py
```

## Load the model
To load our trained model:

1. Go the network folder and unzip the model.zip file

2. Open the ```variables.py``` file and change the ```load_model``` variable to **True**

3. Run the code:
```bash
python main.py
```

