# Using-Machine-Learning-for-Climate-Modelling
Application of 2 Neural Networks to a Slow-Fast Chaotic Dynamical System as a 3 Case Study
    
    Authors: Sergei Soldatenko, Yaromir Angudovich
    Arctic and Antarctic Research Institute, St Petersburg 199397, Russia; sergei.soldatenko@symparico.ca 

This repositorie explores the capabilities of two types of recurrent neural networks, unidirectional and bidirectional long short-term memory networks, for building a surrogate model for a coupled fast-slow dynamic system and predicting its nonlinear chaotic behaviour. The dynamical system in question, comprising two versions of the classical Lorenz model with a small time-scale separation factor, is treated as an atmosphere – ocean research simulator.

## Description of repository files.
testdata.mat - Calculation of the Lorenz’63 model. Lowercase letters x, y, and z for the fast subsystem variables and uppercase letters X, Y, and Z for the slow subsystem variables

weightsLSTM_50_128_3_LSTM.weights.h5 - Stored model weights after machine learning. 3 hidden LSTM layers of 128 units were used. Forecast lead time 0.5 MTU
weightsLSTM_50_128_3_BiLSTM.weights.h5 - 3 hidden LSTM layers and one BiLSTM.  128 units. Forecast lead time 0.5 MTU.

history_50_128_3_LSTM.xlsx - Stored Loss Function results for LSTM model (Learning and Validation)
history_50_128_3_BiLSTM.xlsx - Stored Loss Function results for BiLSTM model (Learning and Validation)


### Figure of loss function (LSTM)
![Loss_function_50_128_3_LSTM](https://github.com/user-attachments/assets/8dcf2d43-35d5-482b-886d-714f6321be16)

### Figure of loss function (BiLSTM)
![Loss_function_50_128_3_BiLSTM](https://github.com/user-attachments/assets/90f0012f-fd72-4ae4-ba57-093859e78831)

### Figure of validation model.
![Validation_50_128_3_BiLSTM](https://github.com/user-attachments/assets/69c437f0-9616-4f1c-8869-66c9b7a15a98)
