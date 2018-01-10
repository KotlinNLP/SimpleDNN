# SimpleDNN [![GitHub version](https://badge.fury.io/gh/KotlinNLP%2FSimpleDNN.svg)](https://badge.fury.io/gh/KotlinNLP%2FSimpleDNN) [![Build Status](https://travis-ci.org/KotlinNLP/SimpleDNN.svg?branch=master)](https://travis-ci.org/KotlinNLP/SimpleDNN)

SimpleDNN is a machine learning lightweight open-source library written in Kotlin whose purpose is to support the 
development of [feed-forward](https://en.wikipedia.org/wiki/Feedforward_neural_network "Feedforward Neural Network") 
and [recurrent](https://en.wikipedia.org/wiki/Recurrent_neural_network "Recurrent Neural Network") Artificial Neural 
Networks.

SimpleDNN is part of [KotlinNLP](http://kotlinnlp.com/ "KotlinNLP") and has been designed to support relevant neural 
network architectures in natural language processing tasks such as pos-tagging, syntactic parsing and named-entity 
recognition.

As it is written in Kotlin it is interoperable with other JVM languages (e.g. Java and Scala). Mathematical operations 
within the library are performed with [jblas](https://github.com/mikiobraun/jblas "jblas"). Different libraries can be 
used instead, although during our experiments it proved to be the fastest. The effort required is minimum as the 
essential mathematical functions are wrapped in a single package and fully tested — saving you valuable time.

### Note

**SimpleDNN does not use the computational graph model and does not perform automatic differentiation of functions.**

In case you are looking for state-of-the-art technology to create sophisticated flexible network architectures, you 
should consider the following libraries: 
[Deeplearning4j](https://github.com/deeplearning4j/deeplearning4j "Deeplearning4j"), 
[mxnet](https://github.com/dmlc/mxnet "mxnet"), 
[DyNet](https://github.com/clab/dynet "DyNet"), 
[Torch](https://github.com/torch/torch7 "Torch"), 
[Keras](https://github.com/fchollet/keras "Keras"), 
[Theano](https://github.com/Theano/Theano "Theano") and 
[TensorFlow](https://github.com/tensorflow/tensorflow "TensorFlow").

If instead a **simpler yet well structured neural network** almost ready to use is what you need, then you are in 
the right place!

## Introduction

Building a basic Neural Network with SimpleDNN does not require much more effort than just configuring a stack of 
neural network layers, with one input layer, any number of hidden layers, and one output layer:

```kotlin
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.core.functionalities.activations.ELU
import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.layers.LayerType

/**
  * Create a fully connected neural network with an input layer with dropout,
  * two hidden layers with ELU activation function
  * and an output one with Softmax activation for classification purpose.
  */
val neuralNetwork = NeuralNetwork(
    LayerConfiguration( // input layer
        size = 784, 
        dropout = 0.25),
    LayerConfiguration( // first hidden layer
        size = 100,
        activationFunction = ELU(),
        connectionType = LayerType.Connection.Feedforward),
    LayerConfiguration( // second hidden layer
        size = 100, 
        ctivationFunction = ELU(),
        connectionType = LayerType.Connection.Feedforward),
    LayerConfiguration( // output layer
        size = 10,
        activationFunction = Softmax(),
        connectionType = LayerType.Connection.Feedforward))
    
neuralNetwork.initialize() // initialize the parameters to random values
```

This library consists of three main building blocks:

- NeuralNetwork: the core module which contains the architectural model.
- NeuralProcessor: it acts (reading) on a NeuralNetwork to compute its training and predictions.
- Optimizer: it optimizes (writing) a NeuralNetwork applying the knowledge given from the errors accumulated during the 
training process.


## Getting Started

### Import with Maven

```xml
<dependency>
    <groupId>com.kotlinnlp</groupId>
    <artifactId>simplednn</artifactId>
    <version>0.1.0</version>
</dependency>
```

### Examples

Try some examples of usage of SimpleDNN running the files in the `examples` folder.

To make the examples working download the datasets 
[here](https://www.dropbox.com/sh/ey4vmajm54xf06v/AADN8nx90WGuOXuzEUY6tbtBa?dl=0 "SimpleDNN examples datasets"), then set their paths 
copying the file `example/config/configuration.yml.example` to `example/config/configuration.yml` and editing it 
properly.   


### Model Serialization

A NeuralNetwork object contains the model as parameters (i.e. *weights* and *bias*) which can be optimized after a 
training process. It provides simple dump() and load() methods to serialize and afterwards read its model.

```kotlin
import java.io.FileInputStream
import java.io.FileOutputStream

fun main(args: Array<String>) {
    
    // Load the network from file
    val inputFilePath = "/path/to/input_net.serialized"
    val fileInputStream = FileInputStream(inputFilePath)
    val neuralNetwork = NeuralNetwork.load(fileInputStream)
    
    // Save the network to file    
    val outputFilePath = "/path/to/output_net.serialized"
    val fileOutputStream = FileOutputStream(outputFilePath)
    neuralNetwork.dump(fileOutputStream)
}
```


## License

This software is released under the terms of the 
[Mozilla Public License, v. 2.0](https://mozilla.org/MPL/2.0/ "Mozilla Public License, v. 2.0")


## Contributions

We greatly appreciate any bug reports and contributions, which can be made by filing an issue or making a pull 
request through the [github page](https://github.com/nlpstep/simplednn "SimpleDNN on GitHub").
