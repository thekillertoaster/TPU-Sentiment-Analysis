# TPU Sentiment Analysis API
This is a simple API written in python which provides an endpoint for sentiment analysis of text.

## About the Model
The model is a simple LSTM model trained on the jigsaw-toxic-comment-classification-challenge dataset. The model is trained on the TPU provided by Google Colab. The model is saved in the models folder.

I've provided the ipynb file for training the model in the root folder of the project, make sure to enable the TPU before running the notebook.

Model Layers:
- ***Embedding Layer:*** This layer converts the input words (represented as integers) into fixed-size vectors of 128 dimensions. It takes 20000 as the maximum number of words to keep based on the word frequency, and 100 as the input length (the maximum number of words in a sequence).
- ***Bidirectional Layer:*** This layer wraps around the LSTM layer, making it bidirectional. A bidirectional LSTM processes the input sequence both in the forward and backward direction, allowing it to capture both past and future context.
- ***LSTM Layer***: This is the actual LSTM (Long Short-Term Memory) layer with 64 units. It is a type of recurrent neural network (RNN) that can learn and remember long-term dependencies in sequential data.
- ***GlobalMaxPool1D Layer:*** This layer applies max-pooling operation along the time dimension (axis 1) of the input data. It is used to extract the most important features from the output of the Bidirectional LSTM layer.
- ***Dense Layer:*** This is a fully connected layer with 50 units and a ReLU (Rectified Linear Unit) activation function. It helps in learning non-linear relationships in the input data.
- ***Dropout Layer:*** This layer randomly sets a fraction (0.1) of the input units to 0 at each update during training time. It helps prevent overfitting by adding regularization to the model.
- ***Dense Layer:*** This is the final layer with 6 units and a sigmoid activation function. It is used to predict the probability of each class. *note:* The sigmoid activation function is typically used for binary classification, as it outputs probabilities between 0 and 1.

The model is compiled with the following specifications:
- ***loss='binary_crossentropy':*** This loss function is appropriate for binary classification problems. It measures the dissimilarity between the true labels and the predicted probabilities.
- ***optimizer='adam':*** Adam (Adaptive Moment Estimation) is an efficient optimization algorithm that adapts the learning rate for each parameter based on the first and second moments of the gradients.
- ***metrics=['accuracy']:*** This indicates that the model will be evaluated using accuracy as the performance metric during training and validation.

## !MORE INFO TO COME SOON!:
- [ ] Requirements.txt
- [ ] Docker / Docker Compose
- [ ] API Documentation