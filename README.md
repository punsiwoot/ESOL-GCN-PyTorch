# ESOL DATASET with GCN
 this is my code when try to imprement and experiment machine learning on graph data structure ESOL dataset

## Train test split
 trainset : 80%
 validationset : 10%
 testset : 10%

## Experiment setup
Activation function : Tanh, Relu
Embedding size : 32, 64, 128
Number of layer : 2, 3, 4
Model Structure : Normal Layer, Pyramid Layer

Normal Layer : a message passing layer with the same size at any layer
Pyramid Layer : a message passing layer with the next layer will have a size devide by 2

From our experiment setup we have a combination of 2*3*3*2 = 36 model to experiment
*** on my training process i will stop when i found that the validation loss is start to increasing(overfitting)

## Summary result of Tunning 
-  Tanh Activation is more suit for capturing the relationship between the Node feature
-  using pyramid structure and normal structure will not affect the performance 
-  At embedding size of 32, 64 and 128 using more number of layer has improve model performance due to model is still underfit to dataset
-  At 128 size of embedding, Normal layer and Tanh Activation function(model 16) we found that the model perform better on 2 number of layer but at further more layer the model starting to overfitting 
- every MSE from testing set is far less than validation set so we consider that the model is overfitting to the testing data
- the model able to capture a relation in dataset but not enough for predict a solubility value

*** in the end Model 16 (best performance) is still have a MSE 0.558 or around 0.746 log(m/l) or around 5.571 mole per lite which is pretty large eror so this result still not good for deploy 

## From this experiment we found that 
For using GCN 
tanh activation function is more affective to this dataset
Using GCN with pyramid structure doesn’t affect to model performance

For father experiment with GCN we suggest that :
	using tanh activation or activation that not cut off a negative value (leaky-relu)
	using more number of layer with multiple embedding size
	using more optimization technique (for example drop out layer)



* ALL Model that i experiment
<img src="/image/AM1.png" alt="Alt text" title="Optional title" width="500" height="500">
<img src="/image/AM2.png" alt="Alt text" title="Optional title" width="500" height="500">

* result of validation set
<img src="/image/validation.png" alt="Alt text" title="Optional title" width="500" height="500">

* result of testing set
<img src="/image/testing.png" alt="Alt text" title="Optional title" width="500" height="500">
 
