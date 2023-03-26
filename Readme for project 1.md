# ML4SCI

This repo contains the .ipynb and .pdfs of the model trained for ML4SCI for GSoC 2023.

# Common Task 1. Electron/photon classification:
         
Task: To classify input as Electron/Photons.

    Dataset :  The below dataset was  provided
                               https://cernbox.cern.ch/index.php/s/AtBT8y4MiQYFcgc (photons)	
 		                   https://cernbox.cern.ch/index.php/s/FbXw3V4XNyYB3o(electrons)
                             
The dataset mainly contained files that were of .hdf5 format. 
The data contained the matrix in the shape of 32,32,2 were the 2 represented the number of channels.
The channels represented hit energy and time.
             



	Approach: --> Firstly i reshaped the matrix into 2,32,32 so that it could be fed into the neural network 

              --> Created a Dataset class that returns the matrix and the label for that matrix denoting the 
              class that matrix belongs to when an index is given under the __getitem__() function 

              --> Then I have made the dataLoader        

              --> The dataset is split into a train and a test set using the function train_test_split() of sklearn. 
              The test size is taken as 10 percent of the given data and the train size is implied that it will be 90 percent.
------------------------------------------------------------------------------------------------------------------------------------------------------------------

    MODEL: → contains 2 conv layers which is each followed by activation ReLU
                 → the last layer contains sigmoid so that it is used for the binary classification
                 →i have 2 fully connected layers 
 ------------------------------------------------------------------------------------------------------------------------------------------------------------------
    HyperParameters:
                → criterion =  nn.BCELoss()
                → optimizer = optim.Adam()
                → number of epochs= 80
                → batch_size = 32
    
------------------------------------------------------------------------------------------------------------------------------------------------------------------
     Results:
       After tuning the hyperparameters as mentioned above i managed to get the ROC AUC score of 80.2.
       
 ![roc curve task 1](https://user-images.githubusercontent.com/102585626/227755120-4c707e71-ee6d-4521-991c-f3dce2f6f451.png)


