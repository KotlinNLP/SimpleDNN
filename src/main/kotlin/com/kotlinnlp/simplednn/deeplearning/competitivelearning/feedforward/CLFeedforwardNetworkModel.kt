/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.competitivelearning.feedforward

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerConfiguration
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.deeplearning.competitivelearning.CLNetworkModel

/**
 * The model of a [CLFeedforwardNetwork].
 *
 * @property numOfClasses the number of classes
 * @property inputSize the size of the input layer
 * @property hiddenSize the size of the hidden layer
 * @property hiddenActivation the activation function of the hidden layer
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
 */
class CLFeedforwardNetworkModel(
  numOfClasses: Int,
  val inputSize: Int,
  val hiddenSize: Int,
  val hiddenActivation: ActivationFunction?,
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = GlorotInitializer()
) : CLNetworkModel(numOfClasses) {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The list of feed-forward sub-networks.
   */
  val networks: List<NeuralNetwork> = this.classes.map {
    NeuralNetwork(
      LayerConfiguration(
        size = this.inputSize,
        type = LayerType.Input.Dense),
      LayerConfiguration(
        size = this.hiddenSize,
        activationFunction = this.hiddenActivation,
        connectionType = LayerType.Connection.Feedforward),
      LayerConfiguration(
        size = this.inputSize,
        activationFunction = null,
        connectionType = LayerType.Connection.Feedforward),
      weightsInitializer = weightsInitializer,
      biasesInitializer = biasesInitializer)
  }
}
