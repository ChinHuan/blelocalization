# Analysis and Discussion

## Stability and Response Rate

A window of sample data is collected to calculate the mean RSSI.  
The larger the window, the more stable the predicted position is. However, larger window leads to a lag in predicted position if the target is moving.
Two tentative methodologies are proposed.

1. Exponential moving average to prioritize more on the current sampled data compared to the data collected previously. (A weighted mean)
2. Outlier detection to reduce the effect of sudden movement. (Take the walking speed into consideration)

## Sample Rate

Highly correlated with the signal strength.
The sample rate might indicate the distance from the sample location to the scanner.
Therefore, we can generate a probability distribution across different scanner location where each probability indicates the probability that the sample location is at the scanner location.

## Standard Deviation of RSSI

The scanners located far away has a low standard deviation because the sample rate is small
The scanners located at middle range usually has high standard deviation because of the attenutation
The scanners located closed to the beacon has a middle standard deviation
