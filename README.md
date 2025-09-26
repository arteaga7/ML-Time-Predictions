# ML-Time-Predictions
The Machine Learning regression problem is addressed for multiple Time Series, implementing the 'Prophet' model, which trains every time series separatly.

**Objective:** To perfom forcast, this is, to make predictions for Time Series to estimate the values of sensed variables. 

## Details
-In folder "notebooks" are the following Jupyter Notebooks:

* "eda.ipynb": Shows how to read the 'parquet' files to obtain the dataset to train the models. The exploratory data analysis is also shown.

* "ml_model.ipynb": Shows how to preprocess the data and how to train 'Prophet' machine learning model to perform forecast for the multiple time series. The metric 'MAE' (mean absolute error) is used to validate the effectiveness of the models. The plots of the time predictions are saved as pictures in folder "../data/processed/prophet_forecasts".

-In folder "data" are the raw and preprocessed data, and some pictures of the predictions.

Finally, script 'main.py' perform all processes previously mentioned (read the 'parquet' files, preprocess the data, train the models and make predicctions).

🛠️ **Python libraries used**: Pandas, Numpy, Matplotlib, Seaborn, Datetime, os, Prophet, Scikit-learn.

## 🚀 Installation
1. Clone this repository:
```
git clone https://github.com/arteaga7/ML-Time-Predictions.git
```
2. Set virtual environment and install dependencies:
```
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```
3. Run "main.py":
```
python3 main.py
```
