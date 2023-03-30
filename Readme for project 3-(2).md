# TASK-3(2): End-to-End Deep Learning Regression for Measurements with the CMS Experiment

Task: To estimate (regress) the mass of the particle based on particle images using the provided dataset. 


Dataset :  The below dataset was  provided
                           https://cernbox.cern.ch/index.php/s/F2rtz6k9TvaWynC	
	                   
This dataset consists of 125x125 image matrices with variables named ieta and iphi. It contains four channels labeled X_jet, which include Track pT, DZ and D0, and ECAL data. For analysis, it was recommended to use at least the ECAL and Track pT channels, with 'am' designated as the target feature.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Approach: 

          --> Firstly i reshaped the matrix into 4,125,125 so that it could be fed into the neural network 

          --> Created a Dataset class that returns the matrix and the label for that matrix denoting the class that matrix belongs to when an index is given under the __getitem__() function 

          --> Then I have made the dataLoader        

          --> The dataset is split into a train and a test set using the function train_test_split() of sklearn to prevent overfitting. The test size is taken as 20 percent of the given data and the train size is implied that it will be 80 percent.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
MODEL: 

          --> The input to the model is a tensor with four channels representing input image data (Track pT, DZ and D0, ECAL).The model has three convolutional layers, each with a kernel size of 3 and padding of 1.
          The number of filters in the three convolutional layers are 16, 32, and 64, respectively.
          
          --> ReLU activation functions are applied after each convolutional and fully connected layer, except for the output layer.
          
          -->The output of the final convolutional layer is passed through two fully connected layers, with 256 and 1 units, respectively.
          
          -->Max pooling with a kernel size of 2 is applied after the first and second convolutional layers to reduce the spatial dimensions of the output.
             The model returns a single scalar value as the prediction, which is obtained by applying a squeeze operation on the final output.
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

HyperParameters:

            → criterion =  nn.MSELoss()
            
            → optimizer = optim.Adam()
            
            → number of epochs= 105
            
            → batch_size = 32
            
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------            
 Results:
 
   After tuning the hyperparameters as mentioned above I managed to get the MSE loss  of the test set around 11 thousand 
   whereas for the train set it was around 7 thousand. So upto some extent the overfitting has been tackled.
   
   ![download (1)](https://user-images.githubusercontent.com/102585626/228789189-4c57f9e3-cc68-423c-b676-cbc57ed03ab5.png)

Fig1: this is the actual vs predicted plot for the test data set which gave MSE error of around 11K.
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

![download (2)](https://user-images.githubusercontent.com/102585626/228789298-1c651000-4b6a-4e6e-8d63-1a2551a7c82b.png)


Fig2: this is the actual vs predicted plot for the train data set which gave MSE error of around 7K.
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


![download](https://user-images.githubusercontent.com/102585626/228789804-1ea8696f-dc30-44de-8f44-08f7085a2141.png)


Fig3: this is the training curve
