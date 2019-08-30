import numpy as np
from numpy.random import seed
import decimal

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.utils import to_categorical
from keras import optimizers
from keras import metrics
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

import pandas as pd

class Group16(QCAlgorithm):
    
#####  Initialization of Algo ####
    def Initialize(self):
  
        #self.Debug("START: Initialize")
        self.SetStartDate(2018,8,8)    # Set Start Date
        self.SetEndDate(2018,8,14)     # Set End Date
        self.SetCash(100000)           # Set Strategy Cash
        self.SetBrokerageModel(BrokerageName.OandaBrokerage, AccountType.Cash) # Set Brokerage Model
        
        ## Target currency pair
        self.currency = "EURUSD"
        ## Other currencies serve as indicators
        self.variable1 = "GBPJPY" 
        self.variable2 = "USDJPY"
        self.variable3 = "GBPUSD"
        
        self.AddForex(self.currency,  Resolution.Hour)
        self.AddForex(self.variable1, Resolution.Hour)
        self.AddForex(self.variable2, Resolution.Hour)
        self.AddForex(self.variable3, Resolution.Hour)
        
        self.long_list =[]
        self.short_list =[]
        self.model1 = LogisticRegression()   ## Model 1: Logistic Regression
        self.model2 =Sequential()            ## Model 2: LSTM Neural Network
        # 15 day moving average              ## Model 3: Moving Average Cross
        
        self.fast = self.EMA("EURUSD", 15, Resolution.Hour)
        # 30 day moving average
        self.slow = self.EMA("EURUSD", 30, Resolution.Hour)
        
        self.previous = None
        self.x=0
        #self.Debug("End: Initialize")

#####  Defining OnData function and Geting the Historical Data  ####
    def OnData(self, data): # This function runs on every resolution of data mentioned
                            
        #self.Debug("START: Ondata")
        currency_data  = self.History([self.currency],  30, Resolution.Hour) # Asking for historical data for the past 48 days
        currency_data1 = self.History([self.variable1], 30, Resolution.Hour)
        currency_data2 = self.History([self.variable2], 30, Resolution.Hour)
        currency_data3 = self.History([self.variable3], 30, Resolution.Hour)
        
        # Checking the length of data
        L= len(currency_data) 
        L1= len(currency_data1)
        L2= len(currency_data2)
        L3= len(currency_data3)

#####  Check condition for required data and prepare X and Y for modeling  ####    
        # Making sure that the data is not empty and three data sources have some length
        # Proceed the prediction only if this condition is met
        if (not currency_data.empty and not currency_data1.empty and not currency_data2.empty and not currency_data3.empty and L == L1 ==L2 == L3 ): 

            data = pd.DataFrame(currency_data.close)  # Get the close prices. Also storing as dataframe
            data1 = pd.DataFrame(currency_data1.close) # dataframes are good for calculating lags and percent change
            data2 = pd.DataFrame(currency_data2.close)
            data3 = pd.DataFrame(currency_data3.close)
            
            # Data Preparation for input to Logistic Regression
            stored = {} # To prepare and store data
            for i in range(11): # For getting 10 lags ... Can be increased if more lags are required
                stored['EURUSD_lag_{}'.format(i)] = data.shift(i).values[:,0].tolist() # creating lags
                stored['GBPJPY_lag_{}'.format(i)] = data1.shift(i).values[:,0].tolist()
                stored['USDJPY_lag_{}'.format(i)] = data2.shift(i).values[:,0].tolist()
                stored['GBPUSD_lag_{}'.format(i)] = data3.shift(i).values[:,0].tolist()

            stored = pd.DataFrame(stored)
            stored = stored.dropna() # drop na values
            stored = stored.reset_index(drop=True)
            
            stored["Y"] = stored["EURUSD_lag_0"].pct_change() # get the percent change of each forex 
            
            for i in range(len(stored)): # loop to make Y as categorical
                if stored.loc[i,"Y"] >= 0:
                    stored.loc[i,"Y"] = "UP"
                else:
                    stored.loc[i,"Y"] = "DOWN"
        
            ### Drop lags that are highly correlated
            corelation = stored.corr()
            #self.Debug("corr is" +str(corelation))
            
            corr_threshold_pos = 0.9
            corr_threshold_neg = -0.9
            
            to_drop_indexes = [] # List of indexes of dropped lags
            for i in range(len(corelation.columns)):
                for j in range(i):
                    if (corr_threshold_pos<corelation.iloc[i, j]) or (corelation.iloc[i, j]<corr_threshold_neg):
                        if j not in to_drop_indexes:
                            to_drop_indexes.append(j) # Add index of dropped lag to list of dropped lag indexes
            
            index_list = [] # List of indexes that we will use
            for i in range(4,44):
                if i not in to_drop_indexes:
                    index_list.append(i)
                
            #self.Debug("All X_data is " +str(stored))    
            X_data = stored.iloc[:,np.r_[index_list]]  # Do not extract Lag0 as Lag0 is the data itself and will not be available during prediction
            
            #self.Debug( "X data is" +str(X_data))
            Y_data = stored["Y"]
            #self.Debug( "Y data is" +str(Y_data))

#####  Build the Logistic Regression model, check the training accuracy and coefficients  ####

            self.model1.fit(X_data,Y_data)
            score1 = self.model1.score(X_data, Y_data)
            self.Debug("Train Accuracy of final model: " + str(score1))
                
            # To get the coefficients from model
            A = pd.DataFrame(X_data.columns)
            B = pd.DataFrame(np.transpose(self.model1.coef_))
            C = pd.concat([A,B], axis = 1)
            #self.Debug("The coefficients are: "+ str(C))
            
            ### Feature Engineering: Only keep variables that are statistically significant
            coef_list = []
            for i in range(B.shape[0]):
                if B.iloc[i,0]>=0.4 or B.iloc[i,0]<=-0.4:  #Set a threshold value of 0.4
                    coef_list.append(i)
            
            # Re-generate the model, with insignificant variables dropped
            if len(coef_list) > 0:
                X_data = X_data.iloc[:,np.r_[coef_list]]
                self.model1.fit(X_data,Y_data)

#####  Prepare data for prediction with logistic regression  ####             
            
            # Prepare test data similar way as earlier
            test = {}
            for i in range(10):
                test['EURUSD_lag_{}'.format(i+1)] = data.shift(i).values[:,0].tolist()
                test['GBPJPY_lag_{}'.format(i+1)] = data1.shift(i).values[:,0].tolist()
                test['USDJPY_lag_{}'.format(i+1)] = data2.shift(i).values[:,0].tolist()
                test['GBPUSD_lag_{}'.format(i+1)] = data3.shift(i).values[:,0].tolist()
            
            index_list_shifted = []
            for i in index_list:
                index_list_shifted.append(i-4)
            
            test = pd.DataFrame(test)
            test = pd.DataFrame(test.iloc[:,np.r_[index_list_shifted]]) 
            if len(coef_list) > 0:
                test = pd.DataFrame(test.iloc[-1,np.r_[coef_list]]) # take the last row of the dataframe(the lastest data)
            else:
                test = pd.DataFrame(test.iloc[-1,:])
            test = pd.DataFrame(np.transpose(test)) # transpose to get in desired model shape

#####  Make Prediction using logistic regression  #### 
    
            output1 = self.model1.predict(test)
            self.Debug("Output from LR model is" + str(output1))
            
#####  Model 2: Build LSTM Model  ####
            data = np.array([currency_data.close])  #Get the close prices and make an array
            self.Debug("Close prices after making an array" + str(data))
            
            # Data Preparation for input to LSTM
            X1 = data[:,0:L-5] # (0 to 25 data)
            self.Debug("X1 is " + str(X1))
            X2 = data[:,1:L-4] # (1 to 26 data)
            self.Debug("X2 is " + str(X2))
            X3 = data[:,2:L-3] # (#2 to 27 data) 
            self.Debug("X3 is " + str(X3))
        
            X= np.concatenate([X1,X2,X3],axis=0) # concatenate to join X1 X2 X3
            self.Debug("X after concatenate:  " + str(X))
            X_data= np.transpose(X) # transpose to get in the form [0,1,2],[1,2,3],[2,3,4],[3,4,5]...
            self.Debug("X after transpose:  " + str(X_data))
        
            Y_data = np.transpose(data[:,3:L-2]) # to get in form [ [3],[4],[5]....
            self.Debug("Y :  " + str(Y_data))
            
            # Normalize the data 
            scaler = MinMaxScaler() 
            scaler.fit(X_data)
            X_data = scaler.transform(X_data)
            #self.Debug("X after transformation is " + str(X_data))
         
            scaler1 = MinMaxScaler()
            scaler1.fit(Y_data)
            Y_data = scaler1.transform(Y_data)
            #self.Debug("Y after transformation is " + str(Y_data))
            
            if self.x==0:  
                ## Only build the model when self.x = 0
                
                # USE TimeSeriesSplit to split data into 5 sequential splits
                tscv = TimeSeriesSplit(n_splits=5)
                
                # Make cells and epochs to be used in grid search
                cells = [100,200]
                epochs  = [100,200]
                
                # creating a dataframe to store final results of cross validation for different combination of cells and epochs
                df = pd.DataFrame(columns= ['cells','epoch','mse'])
                
                # Loop for every combination of cells and epochs. In this setup, 4 combinations of cells and epochs [100, 100] [ 100,200] [200,100] [200,200]
                for i in cells:
                    for j in epochs:
                        
                        cvscores = []
                        # to store CV results
                        # Run the LSTM in loop for every combination of cells an epochs and every train/test split in order to get average mse for each combination
                        for train_index, test_index in tscv.split(X_data):
                            #self.Debug("TRAIN:", train_index, "TEST:", test_index)
                            X_train, X_test = X_data[train_index], X_data[test_index]
                            Y_train, Y_test = Y_data[train_index], Y_data[test_index]
                            
                            #self.Debug("X_train input before reshaping :  " + str(X_train))
                            #self.Debug("X_test is" + str(X_test))
                            #self.Debug("Y input before reshaping:  "+ str(Y_train))
                            #self.Debug("Y_test is" + str(Y_test))
                            
                            #self.Debug ( " X train [0] is " + str (X_train[0]))
                            #self.Debug ( " X train [1] is " + str (X_train[1]))
                            
                            
                            X_train= np.reshape(X_train, (X_train.shape[0],1,X_train.shape[1]))
                            self.Debug("X input to LSTM :  " + str(X_train))
                            X_test= np.reshape(X_test, (X_test.shape[0],1,X_test.shape[1]))
                            self.Debug("Y input to LSTM :  "+ str(Y_train))
                 
                            #self.Debug("START: LSTM Model")
                            #self.Debug(i)
                            #self.Debug(j)
                            model = Sequential()
                            model.add(LSTM(i, input_shape = (1,3), return_sequences = True))
                            model.add(Dropout(0.10))
                            model.add(LSTM(i,return_sequences = True))
                            model.add(LSTM(i))
                            model.add(Dropout(0.10))
                            model.add(Dense(1))
                            model.compile(loss= 'mean_squared_error',optimizer = 'rmsprop', metrics = ['mean_squared_error'])
                            model.fit(X_train,Y_train,epochs=j,verbose=0)
                            #self.Debug("END: LSTM Model")
                            
                            scores = model.evaluate(X_test, Y_test, verbose=0)
                            #self.Debug("%s: %f " % (model.metrics_names[1], scores[1]))
                            cvscores.append(scores[1])
                                
                        MSE= np.mean(cvscores)
                        #self.Debug("MSE" + str(MSE))
                        
                        # Create a dataframe to store output from each combination and append to final results dataframe df
                        df1 = pd.DataFrame({ 'cells': [i], 'epoch': [j], 'mse': [MSE]})
                        self.Debug("Individual run ouput DF1" + str(df1))
                        # Appending individual ouputs to final dataframe for comparison
                        df = df.append(df1) 
                        
                
                self.Debug("Final table of DF"+ str(df))
                
                # Check the optimised values obtained from cross validation
                # This code gives the row which has minimum mse and store the values to O_values
                O_values = df[df['mse']==df['mse'].min()]
                
                
                # Extract the optimised  values of cells and epochs from above row (having min mse)
                O_cells = O_values.iloc[0][0]
                O_epochs = O_values.iloc[0][1]
                
                self.Debug( "O_cells"  + str (O_cells))
                self.Debug( "O_epochs" + str (O_epochs))

                # Build model for whole data:
                # Repeating the model but for optimised cells and epochs
                
                X_data1= np.reshape(X_data, (X_data.shape[0],1,X_data.shape[1]))
                
                #self.Debug("START: Final_LSTM Model")
                
                self.model2.add(LSTM(O_cells, input_shape = (1,3), return_sequences = True))
                self.model2.add(Dropout(0.10))
                self.model2.add(LSTM(O_cells, return_sequences = True))
                self.model2.add(Dropout(0.10))
                self.model2.add(LSTM(O_cells, return_sequences = True))
                self.model2.add(Dropout(0.10))
                self.model2.add(LSTM(O_cells))
                self.model2.add(Dropout(0.10))
                self.model2.add(Dense(1))
                self.model2.add(Activation("softmax"))
                self.model2.compile(loss= 'mean_squared_error',optimizer = 'adam', metrics = ['mean_squared_error'])
                self.model2.fit(X_data1,Y_data,epochs=O_epochs,verbose=0)
                score2 =self.model2.evaluate(X_data1,Y_data,verbose =0)
                score2 = score2[1]
                self.Debug("The accuracy is: " + str(score2))
                #self.Debug("END: Final_LSTM Model")
            
            self.x = 1
            
            # Prepare new data for prediction based on LSTM model
            X1_new = data[:,-3]
            #self.Debug(X1_new)
            X2_new = data[:,-2]
            #self.Debug(X2_new)
            X3_new = data[:,-1]
            #self.Debug(X3_new)

            X_new= np.concatenate([X1_new,X2_new,X3_new],axis=0)
            X_new= np.transpose(X_new)
            #self.Debug(X_new)
            
            scaler = MinMaxScaler() 
            scaler.fit(X_data)
            X_new = scaler.transform([X_new])
            #self.Debug(X_new)
            
            X_new= np.reshape(X_new,(X_new.shape[0],1,X_new.shape[1]))
            #self.Debug(X_new)
            
            # Predicting with the LSTM model
            Predict = self.model2.predict(X_new)
            
            # Needs to inverse transform as we transformed the data for LSTM input
            output2 = scaler1.inverse_transform(Predict)
            #self.Debug("Output from LSTM model is" + str(output2))
            
            # Checking the current price 
            price = currency_data.close[-1]
            self.Debug("Current price is" + str(price))
            
############# Reinforcement Learning:
            # Whenever we are losing money (our equity is smaller than the initial cash set by some amount), retrain the LSTM
            # If not, then we believe the model performance is fine and continue to use the model
            # We have tried using this strategy on the logistic regression model as well, however, the result
            # is not satisfying so we choose to retrain it every not
            
            if self.Portfolio.TotalPortfolioValue<70000:
                self.x = 0
            
            
            # Make decision for trading based on the output from model and currenct price
            # A majority voting process is applied
            # If output (forecast) is UP, we will go long; else, go short.
            # Only one trade at a time and therefore made two lists: "self.long_list" and "self.short_list". 
            # As long as the target currency is in the lists, we restrict further buying
            # Risk and Reward are defined: Exit the market when make certain profits or loss.
            
##### Entry /Exit Conditions for trading  ####
            
            #### Majority Voting
            trend_up = 0
            trend_down = 0
            
            if output1 == "UP":
                trend_up+=1
            else:
                trend_down+=1
            
            if (1-0.02)*output2>=price:    ## a tolerance of 0.02 to take into account the possible noise in the data
                trend_up+=1.2
            else:
                trend_down+=1.2
            
            if self.slow.IsReady and self.previous == None and self.fast.Current.Value > self.slow.Current.Value:
                trend_up+=0.6
            else:
                trend_down+=0.6
                
            ## Initialize parameters to control the exit points
            upper = 1.02       
            lower = 0.99
            ## Initialize parameters to control the amount to long and short
            go_long = 0.95      #[0.85,0.94]
            go_short = -0.95    #[-0.97,-0.85]
            
            if trend_up>=trend_down  and self.currency not in self.long_list and self.currency not in self.short_list :
                # There is an upper trend of the forex
                # In this case, we go long
                # Long(buy) the currency with X% of holding, where X is go_long
                self.SetHoldings(self.currency, go_long)
                    
                #go_long = min(go_long+0.005,0.94)      ## Price will go up--> Forex is profitable--->Higher probability to buy more in the future--->increase buying amount
                                                       ## But we will long the forex with at most 94% of our holdings (go_long<=0.94)
                
                #go_short = min(go_short+0.005,-0.85)   ## At the same time, we short less (decrease the abosolute value of short amount)
                                                       ## But once we decide to short, we will use at least 85% of the holdings (go_short<=-0.85)
        
                self.long_list.append(self.currency)

            if trend_up<trend_down  and self.currency not in self.long_list and self.currency not in self.short_list:
                #self.Debug("output is lesser")
                # There is a downward trend of the Forex
                # In this case, we go short
                # Short(buy) the currency with X% of holding
                self.SetHoldings(self.currency, go_short)

                #go_short = max(go_short-0.005, -0.97)   ## Price will go down---> Forex not profitable---->Higher probability to short more in the future
                                                        ## --> increase the absolute value of amount short
                                                        ## However, we will short with at most 97% of the holdings (go_short>=-0.97)
                                                      
                #go_long = max(0.85, go_long-0.005)      ## At the same time, we will buy less ---> decrease amount long
                                                        ## But once we decide to go long, we will use at least 85% of our current holdings (go_long>=0.85)
                    
                self.short_list.append(self.currency)
                
            if self.currency in self.long_list:
                cost_basis = self.Portfolio[self.currency].AveragePrice
                #self.Debug("cost basis is " +str(cost_basis))

                if  ((price <= float(lower) * float(cost_basis)) or (price >= float(upper) * float(cost_basis))):
                    
                    #self.Debug("SL-TP reached") //stop loss and take profits
                    #If true, exit the market
                    self.SetHoldings(self.currency, 0)
                    
                    self.long_list.remove(self.currency) # clear the forex
                    
                    ## Stop Loss if we lose money
                    #if price<= float(lower) * float(cost_basis):  
                        #lower = min(lower+0.005, 0.99)      ## Once we lose, intuitively, we have to be more cautious of loss, so we increase lower control limit
                                                            ## [just as an example, one usually can't afford to lose anymore if he fails something]
                                                            ## However, we will only exit the market if we make a loss larger than 1%,(lower<=0.99)
        
                        ##upper = max(1.01,upper-0.005)       ## At the same time, losing money means that the target forex is not so profitable
                                                            ## We will compromise by exiting the market with smaller profits (decrease the upper control point)
                                                            ## At the same time, we will 'take profits' only if we have at least made a 1% profit (upper>=1.01)
                     
                    ## Take Profits if we make money
                    ##else:                                   
                        ##upper = min(upper+0.005, 1.1)       ## The Forex is profitable, we are confident that we have higher probability to make money in the future
                                                            ## Therefore, we increase the upper control limit
                                                            ## However, we are always satisfied with 10% profits (upper<=1.1)
                                                                  
                        ##lower = max(0.96,lower-0.005)       ## If we make money, we have stronger capability to afford loss
                                                            ## Therefore, we decrease our lower control limit
                                                            ## But to be safe, we always exit the market if we have a 4% loss (lower>=0.96)
                    self.Debug("squared long")
                        
            if self.currency in self.short_list:
                cost_basis = self.Portfolio[self.currency].AveragePrice
                #self.Debug("cost basis is " +str(cost_basis))
                
                if  ((price <= float(lower) * float(cost_basis)) or (price >= float(upper) * float(cost_basis))):
                    #self.Debug("SL-TP reached")
                    #self.Debug("price is" + str(price))
                    #If true, exit the market
                    self.SetHoldings(self.currency, 0)
                    
                    self.short_list.remove(self.currency)
                    
                    ## Same decision making process as mentioned above
                    ##if price<= float(lower) * float(cost_basis):
                        ##lower = min(lower+0.005, 0.99)
                        ##upper = max(1.01,upper-0.005)
                    ##else:
                        ##upper = min(upper+0.005, 1.1)
                        ##lower = max(0.96,lower-0.005)
                    self.Debug("squared short")
            
            #self.Debug("End OnData")
