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
[PyTorch](https://github.com/pytorch/pytorch "PyTorch"), 
[DyNet](https://github.com/clab/dynet "DyNet"),  
[Keras](https://github.com/fchollet/keras "Keras"),
[TensorFlow](https://github.com/tensorflow/tensorflow "TensorFlow") and
[Deeplearning4j](https://github.com/deeplearning4j/deeplearning4j "Deeplearning4j")

If instead a **simpler yet well structured neural network** almost ready to use is what you need, then you are in 
the right place!

## Introduction

Building a basic Neural Network with SimpleDNN does not require much more effort than just configuring a stack of 
layers, with one input layer, any number of hidden layers, and one output layer:

```kotlin
/**
  * Create a fully connected neural network with an input layer with dropout,
  * two hidden layers with ELU activation function and an output one with 
  * Softmax activation for classification purpose.
  */
val network = StackedLayersParameters(
    LayerInterface( // input layer
        size = 784, 
        dropout = 0.25),
    LayerInterface( // first hidden layer
        size = 100,
        activationFunction = ELU(),
        connectionType = LayerType.Connection.Feedforward),
    LayerInterface( // second hidden layer
        size = 100, 
        ctivationFunction = ELU(),
        connectionType = LayerType.Connection.Feedforward),
    LayerInterface( // output layer
        size = 10,
        activationFunction = Softmax(),
        connectionType = LayerType.Connection.Feedforward))
```

## Getting Started

### Import with Maven

```xml
<dependency>
    <groupId>com.kotlinnlp</groupId>
    <artifactId>simplednn</artifactId>
    <version>0.13.3</version>
</dependency>
```

### Examples

Try some examples of usage of SimpleDNN running the files in the `examples` folder.

To make the examples working download the datasets 
[here](https://www.dropbox.com/sh/ey4vmajm54xf06v/AADN8nx90WGuOXuzEUY6tbtBa?dl=0 "SimpleDNN examples datasets"), then set their paths 
copying the file `example/config/configuration.yml.example` to `example/config/configuration.yml` and editing it 
properly.   

## License

This software is released under the terms of the 
[Mozilla Public License, v. 2.0](https://mozilla.org/MPL/2.0/ "Mozilla Public License, v. 2.0")


## Contributions

We greatly appreciate any bug reports and contributions, which can be made by filing an issue or making a pull 
request through the [github page](https://github.com/KotlinNLP/SimpleDNN "SimpleDNN on GitHub").
