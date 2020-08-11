#importing libraries for analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

#reading the airfoil.csv file using pd.read_csv() method
df=pd.read_csv(r'C:\Users\Ashish Gupta\Downloads\airfoil.csv')
df.head() #to read first five rows of airfoil dataset

#to know the information of data we will use df.info() method
df.info()

#to know the mean, count , mean , First quartile, standard deviation we will use df.describe().T , T is used of transpose of the description matrix 
df.describe().T

#find the pearson correlation using df.corr()
df.corr()
# to check that whether there is a null value in any columns we will use df.isnull().sum()
df.isnull().sum()

#Make the y-vector as the outcome vectore 
y=df['Sound_pressure_level']
df=pd.DataFrame(df) #changing the df as DataFrame Format
df.head()  #to know first five rows of dataset

#creating the x as input dataset 
x=df.iloc[:,:-1] # first : means to select all the rows and second :-1 means to select all the column except the last one which is output columns
x.head()

#Divide the x and y into training and test data by using train_test_split() method in sklearn.model_selection
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42) #test_size=0.25 shows 25% of x and random_state is fixing the seed value
x_train.head()


