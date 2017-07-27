/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.birnn

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
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
 * @property inputType the type of the input layer (Dense, Sparse, SparseBinary)
 * @property inputSize the size of the input layer
 * @property outputSize the size of the output layer
 * @property outputActivation the activation function of the output layer
 * @property outputConnectionType type of recurrent neural network (e.g. LSTM, GRU, CFN, SimpleRNN)
 */
class BiRNN(
  val inputType: LayerType.Input,
  val inputSize: Int,
  val outputSize: Int,
  val outputActivation: ActivationFunction?,
  val outputConnectionType: LayerType.Connection) {

  /**
   * The Recurrent Neural Network to process the sequence left-to-right
   */
  val leftToRightNetwork = NeuralNetwork(
    LayerConfiguration(
      size = this.inputSize,
      inputType = this.inputType),
    LayerConfiguration(
      size = this.outputSize,
      activationFunction = this.outputActivation,
      connectionType = this.outputConnectionType))

  /**
   * The Recurrent Neural Network to process the sequence right-to-left
   */
  val rightToLeftNetwork = NeuralNetwork(
    LayerConfiguration(
      size = this.inputSize,
      inputType = this.inputType),
    LayerConfiguration(
      size = this.outputSize,
      activationFunction = this.outputActivation,
      connectionType = this.outputConnectionType))

  /**
   * Check connection to the output layer.
   */
  init {
    require(this.outputConnectionType.property == LayerType.Property.Recurrent) {
      "required outputConnectionType with Recurrent property"
    }
  }

  /**
   * Initialize the weight of the sub-networks [leftToRightNetwork] and [rightToLeftNetwork] using the default
   * random generator.
   *
   * @return this BiRNN
   */
  fun initialize(): BiRNN {
    this.leftToRightNetwork.initialize()
    this.rightToLeftNetwork.initialize()

    return this
  }
}
