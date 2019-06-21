# Boston Dataset - Jason Hsin

| Name | Date |
|:-------|:---------------|
|Jason Hsin| June 20th, 2019|

-----

### Resources

-----

## Research Question

Objective is to determine the factors (13 attributes) that have higher impact to the pricing of the house. 
In addition, to provide an estimation to price the house owned by customers. 

### Abstract
In this project, the Boston House Prices Dataset was used for the purpose of building regression model and further, for price prediction. Given many features that may have certain impact on the pricing of house, it could be very challenging to come up with a constructive predicting method, given the dimension of the dataset. Machine learning has provided a simple solution to deal with such complex situation. For this case, python based sklearn was adopted to mathematically analyze each feature and their correlation to the listed price. It was found out that natural environment as well as potential employment are the major factors that affected the willingness of a person to buy house, hence affect the price of it more significantly. 

### Methods

For dataset organization and basic statistics, the pandas library was used as it provides very easy handling and access to all location in the dataset. Grouping as well as creating columns can be easily done. For plotting, matplotlib and seaborn were used to give better visualization of the data. Before conducting any analysis, visual interpretation can be sometimes useful in understanding the trend. Finally, the sklearn linear regression model was used to train the dataset and further used for prediction. 


### Results & Discussion
After importing the dataset from the sklearn library, pandas was used to convert it to a data frame for analysis. By using the command describe(), below figure can be shown to provide brief statistics for each feature. The maximum, minimum as well as the mean of the house price can be easily extracted. 


Another method to quickly access the three values is to use the numpy amin, amax, and mean function. 

After the statistics, graphical interpretation of data can be really useful as well to provide us instant understanding of the dataset. A useful tool to plot 2D comparison of all feature is the pairplot function in the seaborn library. The result can be found in the plots folder. It can be seen that features such as RM, DIS has some relative strong correlation to the price. However, this is not enough as to evaluate how crucial they are in affecting the price. Before going into constructing regression model, displot function in seaborn was used. It is seen that the price does follow the normal distribution. 

To finally train the dataset for regression model, the sklearn LinearRegression was used. The dataset was separated into X and Y, with y being the response (price) and X being the rest of the feature. The coefficients were displayed and a for loop as well as if statement were used to quickly identify the features that has stronger impact to the pricing. Take the RM for example, a unit increase in the RM can lead to 3.2 unit increase in the price. Same thing goes with the NOx, CHAS and DIS. 

After fitting fit, the model were than used to predict using the X_test values. The difference between the actual and the predicted were shown in the histogram below. 

Last but not least, Mean square error as well as Root mean square error was calculated to evaluate the performance of the model. From RMSE, it can be concluded that the model is fairly good at predicting the price.  

Finally for a little exercise, when in real life, a real estate agent in the Boston is responsible for giving price estimation of a house. If given a random condition as followed. The resulted predicted value is $376960. 
CROM	ZN	INDUS	CHAS	NOX	RM	AGE	DIS	RAD	TAX	PTRATIO	B	LSTAT
0.03	0	3	0	0.4	5	82	5	2	311	17	396	15

### References
https://github.com/adrianlievano/predict_bostonhousingprices/blob/master/boston_housing.ipynb

-------
