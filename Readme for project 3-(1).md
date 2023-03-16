# Task 3. Vision Transformers for End-to-End Particle Identification with the CMS Experiment:
         
Task: To classify input as Electron/Photons using Vision Transformer.

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

# MODEL: 

# 1. **PatchEmbed**

* **input:** Tensor of the shape **(n,in_chan,widht,height)**

* This class splits the image pixel and then embed them into the embedding dimension

* Basically the approach here is that the image which had 2 channels that is energy channel and time channel this is passed through a conv2d layer of embed dim number of filters. This returns a matrix that has embed_dim of channels

* Since the stride and the filter size of the conv3d was kept to be the same as the patch size, the formula 

$ finaldim = ((initialdim+2*padding-filtersize)/stride)+1$  -----A

* when this formula is applied, the initaildim is taken such that patch size fits it. Therefore it will be some thing like $sqrt(numberOfPatches)* patchDim = imageDim$

* Applying the above equation on equation A we get that final dim = $sqrt(numberFilters)$

* So finally after passing through the conv2d layer the image matrix which was **$(n,2,32,32)$** is now changed into **($n$,$embedDim$,$sqrt(numberFilters)$,$sqrt(numberFilters)$)**

* **output:** At last i have flattened the tensor using .flatten(2) which merges the dim 2 and 3 and then .transpose(1,2) which finally results in the dimesnsion **(n,n_patches,embed_dim)**
                 
 ------------------------------------------------------------------------------------------------------------------------------------------------------------------
 
 # 2. **Attention**

* **input:** Tensor of dimension **(n,n_patches+1,embed_dim)**

* NOTE: Here the dim1 is n_patches +1 becuse the first patch always is the [cls] token which is used to predict the class at the end

* What this class does is that it finds the relation between all the patches/tokens using the concept of attention

* firstly it is passed through a linear layer that outputs 3*dim each corresponding to the qkv that is query, key and value

* after that i split the 3*dim into a (3,n_heads,head_dim)

* Then we matrix multiply the query with key and use the scaling factor as seen in the reseach paper

* after this we use the softmax to the last dimension so that we get the probablity distribution after mutilplying the the value tensor

* that tensor is called as the weighted_avg ,this is then projected by the projection layer and finally we get the output

* **output:** Tensor of the shape (n,n_patches+1,embed_dim)
------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 3. Multilayer Perceptron

* **input:**  Tensor of the shape (n_samples, n_patches + 1, in_features)`

* the forward function of this class contains 2 fully connected layers in which there is one hidden layer and the neurons in the hidden layer is given by the mlp_ratio 

* the second fc layer again converts the tensor back to the input shape and it has 2 dropouts in between inorder to prevent overfitting 
        

* **output:** Tensor of the shape (n_samples, n_patches +1, out_features)
------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 4. Block

* **input:** Tensor of the shape (n_samples, n_patches + 1, in_features)`

* This class contains the residual block that adds itself to the output of the Attention and the MLP classes forward functions.

* First the input is layer normalized and then fed into the Attention and the MLP classes 



* **output:** Tensor of the shape (n_samples, n_patches + 1, in_features)`
------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 5. Vision Transformer

* **input:** Tensor of the shape **(n_samples, in_chans, img_size, img_size)**

* This is the final class that contains all the classes which were defined until now.

* Initially I randomly initialised the cls token to the same size of that of the embed_dim 
   **(1,1,embed_dim)** , and also intialised the position embedding randomly to the shape 
   **(n,n_patches+1,embed_dim)**. Here the +1 is there for the cls token
   
* After this I expanded the the cls token  into the number of trianing examples along the dimension 0

* Then concatenate this into the output that came after passing through the patch_embeds forward function 

* Did the residual adding and added the positional embedding and here the point to be noted is that python broadcasts the positional embedding tensor to the required dimension along the dim0

* Now that the patches/tokens are ready we can proceed to implement the blocks which contains the attnetion and the MLP classes

* In between we have normalisation so that the mean of that dimension becomes 0 and the std deviation becomes 1

* This is observed that it imporoves the trainig

* Now the starting token among the number of patches is taken because that is the cls token and that is used to classify furher.

* the cls token is passed through the linear layer that converts it into the number of classes here i have kept it as 1 and then appplied sigmoid so that i get the probablity of that training example belonging to the class 1


* **output:** Tensor of the shape **(n_samples, 1)**`
------------------------------------------------------------------------------------------------------------------------------------------------------------------
    HyperParameters:
                → criterion =  nn.BCELoss()
                → optimizer = optim.Adam()
                → number of epochs= 80
                → batch_size = 32
    
------------------------------------------------------------------------------------------------------------------------------------------------------------------
     Results:
       After tuning the hyperparameters as mentioned above i managed to get the ROC AUC score of 75.6. Actually the accuracy could be more better but then I have trained the model on less data due to computational limitaion. Therefore the ROC AUC value is a bit less than that of the CNN model.


------------------------------------------------------------------------------------------------------------------------------------------------------------------
