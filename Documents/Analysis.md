# Analysis and Discussion

## Ideas

### **Stability and Response Rate**

A window of sample data is collected to calculate the mean RSSI.  
The larger the window, the more stable the predicted position is. However, larger window leads to a lag in predicted position if the target is moving.
Two tentative methodologies are proposed.

1. `Exponential moving average` to prioritize more on the current sampled data compared to the data collected previously. (A weighted mean)
2. `Outlier detection` to reduce the effect of sudden movement. (Take the walking speed into consideration)

### **Sample Rate**

Highly correlated with the signal strength.
The sample rate might indicate the distance from the sample location to the scanner.
Therefore, we can generate a probability distribution across different scanner location where each probability indicates the probability that the sample location is at the scanner location.

### **Standard Deviation of RSSI**

The scanners located far away has a low standard deviation because the sample rate is small  
The scanners located at middle range usually has high standard deviation because of the attenutation  
The scanners located closed to the beacon has a middle standard deviation  

### **Large Location Deviation Predicted by Classifier**

Could use `k-NN` to limit the false predictions.

---

## Decisions

- GroupBy or Rolling  
  Decision: `Rolling`
  - Advantages
    - For small amount of data, rolling is better as more data is remained
    - Group by reduces too much data

  - Disadvantages
    - Rolling will roll over the data in different collection data

  - Note
    - Not yet tested on big data

- Forward fill or not
- Imputation with 0 or -100  
  **MinMaxScaling**  
  Decision: `-100`
  - Advantages
    - After applying MinMaxScaling, -100 will become 0

- Scaling (No scaling, MinMaxScaling, StandardScaling ...)

- Classification or Regression

---

## Planning

### **Preprocessing**

#### Time Series Analysis

- Moving average
- Exponential smoothing
- ARIMA

### **Model Selection**

- Regression (Linear, Ridge, Logistic)
- Classification (k-NN, SVM)

- Multi-Layer Perceptron
- Convolutional Neural Network
- Recurrent Neural Network (LSTM)

### **Error Analysis**

#### Separability

- Analyse histogram separation
  - For each scanner, analyse the distribution of RSSI for each location

#### Visualisation

- Identify possible visualisation to explore the data

---
