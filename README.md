# Montreal Local TV Channels
The implementation is about predicting market-share about some some episodes in Montreal Local TV Channels between 2016-2019

## Dataset
These data belong to the Montreal Local TV channels, which contains several features including: Name of episode, data, time, etc.

Dataset has around 12 feature columns and about 600k samples.

## Libs Used
**Pandas**: Used for data manipulation

**Numpy**: Powerful library for numerical computation

**Matplotlib**: Useful in plotting and data visualisation

**Scikit-learn**: Powerful library with implementation most of the machine learning library and performance metric

**joblib**: for saving the trained model.

## Files
**Montreal_TV.py**: A python script for model training, saving the model and generate prediction.csv

**Montreal_TV.ipynb**: A jupyter notebook with additional data visualization and analysis.


## Model
**Random Forest Regressor**: After testing several models including: SGDRegressor, Lasso, Decision Tree, Random Forest it is found that RF is best suitable for this task.

## Result
Metrics were MAE and R-Squared. 

| Metric  | Train         | Test          |  
|---------| ------------- | ------------- |
|R-Squared| 0.91          | 0.88          |
|MAE      | 0.96          | 1.05          |

## Further Plans
**1-Scikit-learn pipelining**: by using Scikit-learn pipelining, data cleanning can be done better and code readability enhances.

**2-Apache Spark Distributed Tools**: by using apache spark speed of the training phase may be increased by help of distributed processing.
