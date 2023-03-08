# Common Task 2. Quark/Gluon classification:

Task: To classify input as Electron/Photons.

    Datasets: https://cernbox.cern.ch/index.php/s/hqz8zE7oxyPjvsL 

This dataset was of the file format  .test.snappy.parquet extension
It contains a matrix of shape (3,) in which each element was an array of 125 elements and that 125 elements had 125 elements. 
The dataset also contained the m0 and pt values along with the target which was binary due to binary classification problem statement.


    Approach:
	Created a dataframe which has the matrix of images of shape  3,125,125 and pt m0 and label y. 


	Then I created the class dataset and dataloader and then split the data into training 
	and testing sets so that I can validate my model. 80% of the data were used for the 
	training and the remaining 20% was used for the test purpose.
------------------------------------------------------------------------------------------------------------------------------------------------------------------

    MODEL:
	Then  the class of architecture which contains the architecture from nn.module was made. 


	Basically it had 3 conv2d layers each followed by activation ReLU and the maxpool2d layers. 
	At the end I have kept 2 fully connected layers and applied sigmoid to the final layer so 
	that it gives us the probability of belonging to class 1.
------------------------------------------------------------------------------------------------------------------------------------------------------------------

	HyperParameters:
                → criterion =  nn.CrossEntropyLoss()
                → optimizer = optim.Adam()
                → number of epochs= 150
                → batch_size = 32
 ------------------------------------------------------------------------------------------------------------------------------------------------------------------

               Results: The ROC AUC Score on the test data was 69.9
