# Heart-Rate-Analysis-and-Prediction-Models

#### Link To Data: https://www.kaggle.com/datasets/saurav9786/heart-rate-prediction

#### Related Project of Mine: Heart Attack Physiological Factors EDA and Prediction Models
Python EDA looking at the physiological factors contributing to heart attacks for over 300 patients. After looking at multiple ML predictive models, able to predict the likelihood of a heart attack with up to 92%
#### Link to my previous project: https://github.com/petermartens98/Heart-Attack-Analysis-and-Predictive-Models 


## Objective
The objective is to build a regressor model which can predict the heart rate of an individual.

## Project Description
This code defines a neural network model using Keras to predict the heart rate of an individual based on signals measured using ECG. It uses a training dataset to train the model and a testing dataset to evaluate its performance.

The first two functions, plot_corr_heatmap and plot_distribution, are helper functions to visualize the correlation matrix and distribution of the dataset respectively. The dataset is then split into train and test sets, and the target variable is set to "HR", which represents the heart rate.

The input data is normalized using StandardScaler to scale the values to have zero mean and unit variance. The model architecture consists of three fully connected (dense) layers with ReLU activation function, with each layer followed by a dropout layer to prevent overfitting. The output layer is a single node with linear activation function, as this is a regression problem.

The model is then compiled using the Adam optimizer and mean squared error (MSE) as the loss function, with mean absolute error (MAE) as the metric to monitor during training. The model is trained for 100 epochs with a batch size of 32 and a validation split of 0.2.

The performance of the model is evaluated on the test set using the evaluate function, which computes the test loss and MAE. The y_pred variable is used to store the predicted values for the test set, and a scatter plot of actual versus predicted values is created to visualize the model's performance.

The final section of the code uses scikit-learn's GridSearchCV to perform hyperparameter tuning on the model. It creates a parameter grid with different combinations of activation functions, dropout rates, and weight decay values, and searches over these hyperparameters to find the combination that results in the best performance. The create_model function is defined to create a Keras model with the given hyperparameters, and the KerasRegressor object is used to wrap this model for use in scikit-learn.

## Dataset
The data comprises various attributes taken from signals measured using ECG recorded for different individuals having different heart rates at the time the measurement was taken. These various features contribute to the heart rate at the given instant of time for the individual.

There are total of 6 CSV files with the names as follows:
time_domain_features_train.csv - This file contains all time domain features of heart rate for training data
frequency_domain_features_train.csv - This file contains all frequency domain features of heart rate for training data
heart_rate_non_linear_features_train.csv - This file contains all non linear features of heart rate for training data

time_domain_features_test.csv - This file contains all time domain features of heart rate for testing data
frequency_domain_features_test.csv - This file contains all frequency domain features of heart rate for testing data
heart_rate_non_linear_features_test.csv - This file contains all non linear features of heart rate for testing data

## Variables

Following is the data dictionary for the features you will come across in the files mentioned:

MEAN_RR - Mean of RR intervals

MEDIAN_RR - Median of RR intervals

SDRR - Standard deviation of RR intervals

RMSSD - Root mean square of successive RR interval differences

SDSD - Standard deviation of successive RR interval differences

SDRR_RMSSD - Ratio of SDRR / RMSSD

pNN25 - Percentage of successive RR intervals that differ by more than 25 ms

pNN50 - Percentage of successive RR intervals that differ by more than 50 ms

KURT - Kurtosis of distribution of successive RR intervals / "tailedness"

SKEW - Skew of distribution of successive RR intervals

MEAN_REL_RR - Mean of relative RR intervals

MEDIAN_REL_RR - Median of relative RR intervals

SDRR_REL_RR - Standard deviation of relative RR intervals

RMSSD_REL_RR - Root mean square of successive relative RR interval differences

SDSD_REL_RR - Standard deviation of successive relative RR interval differences

SDRR_RMSSD_REL_RR - Ratio of SDRR/RMSSD for relative RR interval differences

KURT_REL_RR - Kurtosis of distribution of relative RR intervals

SKEW_REL_RR - Skewness of distribution of relative RR intervals

uuid - Unique ID for each patient

VLF - Absolute power of the very low frequency band (0.0033 - 0.04 Hz)

VLF_PCT - Principal component transform of VLF

LF - Absolute power of the low frequency band (0.04 - 0.15 Hz)

LF_PCT - Principal component transform of LF

LF_NU - Absolute power of the low frequency band in normal units

HF - Absolute power of the high frequency band (0.15 - 0.4 Hz)

HF_PCT - Principal component transform of HF

HF_NU - Absolute power of the highest frequency band in normal units

TP - Total power of RR intervals

LF_HF - Ratio of LF to HF

HF_LF - Ratio of HF to LF

SD1 - Poincaré plot standard deviation perpendicular to the line of identity

SD2 - Poincaré plot standard deviation along the line of identity

Sampen - sample entropy which measures the regularity and complexity of a time series

higuci - higuci fractal dimension of heartrate

datasetId - ID of the whole dataset

condition - condition of the patient at the time the data was recorded

HR - Heart rate of the patient at the time of data recorded
