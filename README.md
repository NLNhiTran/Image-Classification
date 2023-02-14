# Image Classification of Sign Language Letters 
## Machine Learning Challenge 2022

In this image classification task two machine learning methods were adapted to perform on a dataset containing sign language images. Convolutional Neural Networks (CNN), as well as Support Vector Machine (SVM) algorithms, have been selected in accordance with the substantiations of tasks 1.

### Model 1

The image classification problem was approached using the supervised learning algorithm Support Vector Machine, which has been found to be a competitive machine learning technique in pixel-based image classification and multi-dimensional datasets since it can produce satisfactory results and a memory-efficient model. 

Data was initially loaded in two separate train and test sets. It was learned that the train data set has a total of 27,455 grayscale images of size 28 x 28. 

It is needed to normalize the features with high variances hence the pixelated images were normalized by  converting the original individual data values from the range [0,255] to [0,1]. 

Next, the data labels are checked for class imbalance. The data is relatively balanced across the 24 classes. 

![image](https://user-images.githubusercontent.com/117458345/218732068-ce205919-54fb-47a5-8b94-a2bdc5e1f879.png)

70% of data was used for training and 30% was used to tune the hyperparameters. Sigmoid, RBF, and Polynomial kernels were tested to optimize the decision boundaries since they are most useful in multidimensional and non-linear separation problems. 

![image](https://user-images.githubusercontent.com/117458345/218732765-c550c6f0-429b-4d6c-9380-2aba54098aca.png)

### Model 2

For the second model a deep neural network was constructed using a convolutional neural network (CNN). Since the input images are in grayscale, there are only 2 dimensions and thus a 2D-CNN was constructed. The constructed model consists of two convolutional layers. Both layers are followed by a max-pooling layer to reduce the number of elements in the feature map and a ReLU activation function to generate linearity and reduce vanishing gradients. The last two parts of the CNN consist of fully connected linear layers that enable the final prediction of a class. Since this is a multi-classification problem and not a binary problem the final activation function is the softmax function. The results of the following are CNN compositions that can be seen in appendix 3.

In order to create the best multi-classification algorithm with this model composition, the negative log-likelihood error function was used as a loss function. It is possible to generate similar results by removing the LogSoftmax from the model composition and using the Cross-Entropy Loss as the error function. 

Before training a validation and training set were created. This was done by using the random_split function from PyTorch. As the dataset is fairly large, the validation set was chosen to be 10% of the total dataset used for training.  

After the implementation of the final model on the test dataset, an overall accuracy of 93% was achieved. The most difficult classes to classify are 19, 17, and 13. These classes have similarities to other classes and may therefore be predicted as if they are of a different class. E.g., 13 is similar to 10, 17 to 3, and 19 to 6. 

![image](https://user-images.githubusercontent.com/117458345/218732917-d6654e56-4ee4-4c4a-a7bd-e2cdbca52c29.png)

### Evaluation

When comparing the models to one another, it can be stated that the convolutional neural network approach resulted in a considerably stronger test set result. With a final test set score of 93%, it scored 15 percentage points higher than the 78% test set score of the support vector machine approach. On top of that, model 2 provided solid accuracy scores on the individual labels across the board, whereas model 1 plainly failed to classify label 17 accurately, and had difficulties with multiple other labels (e.g., labels 10, 18, 20, 22). 

When it comes to the training time of model 2, approximately one minute was needed to run the code to get a result. Due to manual hyperparameter optimization, the tuning process took similar amounts of time for each configuration. As for model 1, the training time took approximately 19.49 seconds and the tuning job took almost 5 minutes to iterate over the different combinations of parameters mentioned above. 

### Advanced predictions

For the second task it was required to use an already existing classifier or make another one to identify a series of n sign language letters in an image. The provided pictures were of size 28x200 and the above-mentioned variable n can take values in the range from 1 to 5 randomly. Moreover, the image background is noisy. 

After importing packages and datasets, the loaded images were looped over. In the inner loop, different thresholding parameter values were chosen for the pixel neighborhood and threshold finetune to create bounding boxes. Each image was first changed to grayscale (0-255), then the gaussian blur was applied and finally, the whole image was thresholded using the cv2.adaptiveThreshold method which is the function that automatically chooses a correct threshold for images. Above operations allowed for use of the findContours function from the cv2 library which led to getting the contours of the nested images. Finally, the results were sorted to obtain the right order. 

After conducting image processing it was possible to utilize the trained model to make predictions. The convolutional neural network (CNN) model was chosen as it was a better predictor in the previous task. Before looping over the obtained images, bounding boxes in the images were defined. Also, it was double-checked that bounding boxes had a minimum size that could display a sign language image. Furthermore, since the size of the bounding box can be less than 28x28, the middle of the width was found and taken as the center point together with the full height. This is necessary to be able to pass the image through the layers of the pre-trained CNN model. After getting the right format, the pictures were passed through the model, and by using the .argmax function the ones with the biggest probabilities were chosen. Finally, zeros were added in the predicted classes with only one digit. Extracted predictions were added to the final list of predictions and then saved in the .csv file. 

The top five predictions were picked out for each image based on various thresholding parameters that give different bounding boxes thus different sub-images. Based on the predictions made, it is noticed that the overall performance of the model is relatively lower compared to that in task 1 where the model has to run through each single image row by row. The model appears to produce lower accuracy scores for some particular classes such as 10 and 24. It is also learned that class 05 is wrongly predicted quite often. This problem could be fixed by either enhancing further the image quality, modifying the parameter choices to retrieve more accurate bounding boxes, or applying data augmentation to the dataset for the model training. In general, the model performs reasonably well as a result of the sanity check (Appendix 5). It made a 100% accurate prediction for the first image, and partially correct predictions for the second and the last image of the dataset due to the reasons discussed earlier. 

![image](https://user-images.githubusercontent.com/117458345/218733034-e6b5392e-0996-4c2a-bd48-3a7126d20fa4.png)

