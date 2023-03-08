# Further plans in the ViT Model:

* As the project is based on Vision Transformer for binary classification, one of the future plans is to further optimize the model's performance. 
We can explore ways to fine-tune the hyperparameters and  increase the dataset size.

* Have thought of the different modifications which I would perform.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
→ As stated in the research paper "Attention is all you need" , The time complexity of the model is O(m^2) . 
That is the time complexity of the model is quadratic and while I trained the model I also felt the same, 
It took lots of time even after using a GPU. By using filters with stride I am expecting better results because in that 
case the number of filters is reduced and the time is saved.

 Time plays a crucial role so that we can experiment with the model and tune the hyperparameters. 
 In the case of CNN network I could manage to train the model in ⅕ th time of that taken by the ViT the reason being the above.

→ Use a hybrid between the CNN and the Vision Transformer: What I meant by a hybrid is that I will run a few layers 
of Conv2d on the image first and then convert it into patch embed. Here in the current model the number of channels 
of the input was 2 and converted into the embed dimension directly. What I think is that using a couple of Convolutional 
layers in between would perform better.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


* Ensemble: To increase the performance of the model I have thought to implement the ensemble between multiple models. 

* As I read the research paper I came to know that ViT models can outcast the CNN models only with a huge amount of data. 
  When there is less data CNN works better than ViT and vice versa when there is more data. 
  So I want to extend the model to more training examples. One more idea that I have is that to use Data Augmentation technique.

* Talking about the validation of the model I have just split the data into a train and test set. 
  I want to extend this to K-fold cross validation so that I don't overfit the model and can fine tune the hyperparameters.

*Additionally, we could investigate ways to apply the model to other similar problems or tasks. We can extend the vision transformer to the particle classification which is used in the Compact Muon Solenoid (CMS) experiments in the Large Hadron Collider (LHC).
