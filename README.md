# Store Demand Forecasting: Classical vs Machine Learning vs Deep Learning

Course project for **IE 590MLA: Machine Learning and Its Applications**, Purdue University. Team of five (Team 6), instructor Dr. Congfang Huang.

## Question

Which model type forecasts daily store-item demand best, and how much does the choice matter for aligning production, inventory and logistics?

## Data

Kaggle Store Item Demand Forecasting Challenge: 913,000 daily sales records from 2013-01-01 to 2017-12-31, covering 10 stores and 50 items, with no missing values or duplicates. Mean daily sales per store-item is about 52 units (maximum 231).

The Kaggle test file has 45,000 rows for 2018-01-01 to 2018-03-31 and no sales column, so it cannot be scored. All reported errors come from holding out part of the training file.

Preprocessing: duplicate and null checks, date split into day, month and year features, chronological sort, 80/20 train/test split.

## Models

| Model | Setup |
|---|---|
| XGBoost | Store, item, day, month, year features. 200 trees, learning rate 0.1, max depth 9, subsample 0.6, gamma 0. Inputs and target standardized. Split is random and shuffled (random_state 8), not chronological. |
| ANN | Dense layers 128, 64, 32, 1 with ReLU, Adam (learning rate 0.001), MSE loss, 30 epochs, batch 32, 7-day input window, min-max scaling |
| CNN | 7 time steps by 6 features. Conv1D (64 filters, kernel 3), max pooling, Conv1D (32 filters, kernel 2), dense 32, dense 1. 30 epochs, batch 32, min-max scaling, chronological split |
| Random Forest | 100 trees, max depth 10, sqrt features, on store, item and year. Raw (unscaled) sales |
| ARIMA(5,1,3) | Fitted to total daily sales across all stores and items, standardized, chronological split |

Stack: Python, pandas, scikit-learn, XGBoost, TensorFlow/Keras, statsmodels.

## Results as reported

Train and test error from the team report and presentation. The models were scored on different scales, so the columns are not comparable across rows.

| Model | Scale of the metric | Test MSE | Test MAE |
|---|---|---|---|
| Random Forest | Raw sales units | 386.58 | 14.99 |
| XGBoost | Standardized | 0.091 | 0.2264 |
| ANN | Min-max scaled | 0.0158 | 0.1021 |
| CNN | Min-max scaled | 0.0157 | 0.1029 |
| ARIMA | Standardized daily totals | 10.35 | 2.895 |

The report concludes that the ANN and CNN were best, XGBoost was competitive, and Random Forest and ARIMA were weaker.

## Checking the comparison on one scale

Converting the reported errors to sales units, using the training standard deviation (about 28.8) and the min-max range (231), gives these approximate test MAEs: XGBoost about 6.5, Random Forest about 15.0, CNN about 23.8. Predicting the training mean for 2017 would give an MSE of about 1,063 units squared, against about 838 for the CNN. On a common scale the ranking therefore does not favor the neural networks, and the XGBoost figure is likely optimistic because its split was shuffled across all years. The conversions are approximate.

## Next steps

- Score every model on the same chronological hold-out (for example 2017) in raw sales units.
- Add a seasonal naive baseline, and lag and holiday features.
- Try LSTM and GRU models and probabilistic forecasts, as proposed in the report.

## References

Code and project materials cited in the team report: https://github.com/aravinda-1402/Store-Demand-Forecasting-using-Time-Series-and-Neural-Networks
