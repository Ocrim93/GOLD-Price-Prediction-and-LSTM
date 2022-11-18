# Description: this program uses an artificial recurrent network called Long Short Term Memory(LSTM)
             # to predict the closign stock price of a corporation (Apple Inc) using the past 60 day stock price
import datetime
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt             
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
plt.style.use("fivethirtyeight")


# Get the stock quote
start = datetime.datetime(2012,1,1)
end = datetime.datetime(2021,2,17)

df = web.DataReader('GOLD',data_source="yahoo",start=start, end=end)
#print(df)
# Get the number of rows and colums in the data set

#print(df.shape)
#Visualize the closing price hystory 
plt.figure(figsize=(16,8))
plt.title('Close Price Hisory')
plt.plot(df['Close'])
plt.xlabel("Date",fontsize=18)
plt.ylabel('Close Price USD ($)',fontsize=18)
#plt.show()

# Create a new dataframe with only the Close Column
data = df.filter(['Close'])  # 2 columns, date and close price 
dataset = data.values # set with only close price
training_data_len = math.ceil(len(dataset) * 0.8) # train the 80% of our dataset

#Scale the data 

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
print(scaled_data)


#Create the training data set 
#Create the scaled training data set
train_data = scaled_data[0:training_data_len]
#print(len(train_data))

#Split the data into x_train and y_train sets
x_train = []
y_train = []

for i in range(60,len(train_data)):
	x_train.append(train_data[i-60:i,0])
	y_train.append(train_data[i,0])


#Convert the x_train and y_train to numpy array

x_train , y_train = np.array(x_train) , np.array(y_train)

# LSTM model expects 3D array
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
print(x_train.shape)
# Build LSTM model 
#element of RNN
model = Sequential()

# units: the dimension of output space (number of neurons)
# input_shape: the shape of the training set(number of time step and feature)
#return_sequences: True or False, determines whether to return the last output in the output sequence or the full sequence

model.add(LSTM(units=50,return_sequences = True, input_shape = (x_train.shape[1],1)))

# Droupout layer is a type of regularization technique which is used to prevent overfitting, 
# but it may also increase training time in some cases
model.add(LSTM(units = 50,return_sequences=False))
model.add(Dense(25))#number of neurons 
model.add(Dense(1))

# Compile the model 

model.compile(optimizer='adam',loss='mean_squared_error')

# Train the model 
# epochs are the number of times we pass the data into the neural network

model.fit(x_train,y_train,batch_size=1,epochs=1)

#Create the testing data set

#Create a new array containing scales values from index 1543 to 2003
test_data = scaled_data[training_data_len-60:,:]

# Create the data sets x_test y_test
x_test = []
y_test = dataset[training_data_len:]    # training_data_len = 1603
for i in range(60,len(test_data)):
	x_test.append(test_data[i-60:i,0])


# Convert the data ti a numpy array
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))	

# Get the model predicted price values 

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared Error rmse

rmse = np.sqrt(np.mean((predictions - y_test)**2))
print(rmse)
# Plot the data 

train= data[:training_data_len]
valid = data[training_data_len:]
valid["Predictions"] = predictions
# Visualize the data
plt.figure(figsize=(16,8))
plt.title("Model")
plt.xlabel("Data", fontsize=18)
plt.ylabel("Close Price USD $",fontsize=18)
plt.plot(train["Close"],color='red')
plt.plot(valid["Close"],color='blue')
plt.plot(valid["Predictions"],color='orange')
plt.legend(["Train","Val","Predictions"],loc = "lower right")
plt.show()

print(valid)















