/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.birnn

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerConfiguration
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork

/**
 * Bidirectional Recursive Neural Network (BiRNN)
 *
 * The class contains the sub-networks which constitute a BiRNN.
 *
 * Reference:
 * Mike Schuster and Kuldip K. Paliwal. - Bidirectional recurrent neural networks
 *
 * @property inputLayerSize size of the input layer
 * @property hiddenLayerSize size of the hidden layer
 * @property hiddenLayerActivation activation function of the hidden layer
 * @property hiddenLayerConnectionType type of recurrent neural network (e.g. LSTM, GRU, CFN, SimpleRNN)
 * @property outputLayerSize size of the output layer
 * @property outputLayerActivation activation function of the output layer (could be null)
 */
class BiRNN(
  val inputLayerSize: Int,
  val hiddenLayerSize: Int,
  val hiddenLayerActivation: ActivationFunction?,
  val hiddenLayerConnectionType: LayerType.Connection = LayerType.Connection.GRU,
  val outputLayerSize: Int,
  val outputLayerActivation: ActivationFunction?) {

  /**
   * The Recurrent Neural Network to process the sequence left-to-right
   */
  val leftToRightNetwork: NeuralNetwork

  /**
   * The Recurrent Neural Network to process the sequence right-to-left
   */
  val rightToLeftNetwork: NeuralNetwork

  /**
   * The Feedforward Neural Network that combine the bidirectional output
   * using a linear transformation followed a the non-linear activation function.
   */
  val outputNetwork: NeuralNetwork

  init {
    require(hiddenLayerConnectionType.property == LayerType.Property.Recurrent) {
      "required hiddenLayerConnectionType with Recurrent property"
    }

    this.leftToRightNetwork = NeuralNetwork(
      LayerConfiguration(size = this.inputLayerSize),
      LayerConfiguration(
        size = this.hiddenLayerSize,
        activationFunction = this.hiddenLayerActivation,
        connectionType = this.hiddenLayerConnectionType))

    this.rightToLeftNetwork = NeuralNetwork(
      LayerConfiguration(size = this.inputLayerSize),
      LayerConfiguration(
        size = this.hiddenLayerSize,
        activationFunction = this.hiddenLayerActivation,
        connectionType = this.hiddenLayerConnectionType))

    this.outputNetwork = NeuralNetwork(
      LayerConfiguration(size = this.hiddenLayerSize * 2),
      LayerConfiguration(
        size = this.hiddenLayerSize,
        activationFunction = Tanh(),
        connectionType = LayerType.Connection.Feedforward),
      LayerConfiguration(
        size = this.outputLayerSize,
        activationFunction = this.outputLayerActivation,
        connectionType = LayerType.Connection.Feedforward))
  }

  /**
   * Initialize the weight of the sub-networks [leftToRightNetwork],
   * [rightToLeftNetwork], [outputNetwork] using the default random generator.
   *
   * @return this BiRNN
   */
  fun initialize(): BiRNN {
    this.leftToRightNetwork.initialize()
    this.rightToLeftNetwork.initialize()
    this.outputNetwork.initialize()

    return this
  }
}
