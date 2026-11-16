# Store Demand Forecasting: Classical vs Machine Learning vs Deep Learning

Course project for **IE 590MLA: Machine Learning and Its Applications**, Purdue University. Team of five (Team 6), instructor Dr. Congfang Huang.

## Question

Which model type forecasts retail demand best, and how much does model choice matter for aligning production, inventory and logistics?

## Data

Kaggle Store Item Demand Forecasting Challenge. Daily unit sales by store and item, about 913K training rows (2013 to 2017) and 45K rows to predict for 2018.

Preprocessing: duplicate and null checks, date standardization (day, month, year features), chronological sort, 80/20 time-based train/test split.

## Models

| Model | Setup |
|---|---|
| XGBoost | Tuned n_estimators, learning rate, max depth, subsample, colsample_bytree |
| ANN | Dense 128, 64, 32, 1. ReLU, Adam (0.001), MSE loss, 30 epochs, batch 32, 7-day window, MinMax scaling |
| CNN | Two Conv1D layers (64 and 32 filters), max pooling, dense head, 7-day window, 30 epochs |
| Random Forest | 100 trees, max depth 10, features: store, item, year |
| ARIMA | Classical time-series baseline |

## Findings

- ANN and CNN delivered the highest accuracy.
- XGBoost was reasonable but behind the neural networks.
- ARIMA beat Random Forest but struggled with non-linear patterns.
- Random Forest did poorly because it does not capture temporal dependence.
- Train and test errors were close, which suggests stable models.

## Tools

Python, XGBoost, scikit-learn, Keras/TensorFlow, statsmodels

## Team

Adharsh Rajasekar, Anusha Prabhakaran, Duna Areny Molne, Gautham Vepa, Sriraga Devasani
