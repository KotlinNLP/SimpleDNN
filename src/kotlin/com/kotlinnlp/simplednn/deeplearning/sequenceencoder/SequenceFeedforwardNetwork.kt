/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.sequenceencoder

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.layers.LayerConfiguration
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork

/**
 * Sequence Feedforward Network.
 *
 * It encodes a sequence of arrays into another sequence of arrays through a single Feedforward layer.
 *
 * @property inputType the type of the input layer (Dense, Sparse, SparseBinary)
 * @property inputSize the size of the input layer
 * @property outputSize the size of the output layer
 * @property outputActivation the activation function of the output layer (could be null)
 */
class SequenceFeedforwardNetwork(
  val inputType: LayerType.Input,
  val inputSize: Int,
  val outputSize: Int,
  val outputActivation: ActivationFunction?) {

  /**
   * The Feedforward Neural Network which encodes each input array into another array.
   */
  val network: NeuralNetwork = NeuralNetwork(
    LayerConfiguration(
      size = this.inputSize,
      inputType = this.inputType),
    LayerConfiguration(
      size = this.outputSize,
      activationFunction = this.outputActivation,
      connectionType = LayerType.Connection.Feedforward))

  /**
   * Initialize the weight of the network using the default random generator.
   *
   * @return this BiRNN
   */
  fun initialize(): SequenceFeedforwardNetwork {
    this.network.initialize()
    return this
  }
}
