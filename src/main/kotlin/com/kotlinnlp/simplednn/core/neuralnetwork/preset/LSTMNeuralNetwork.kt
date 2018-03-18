/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralnetwork.preset

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerConfiguration
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork

/**
 * The Long-Short Term Memory Recurrent Network factory.
 */
object LSTMNeuralNetwork {

  /**
   * @param inputSize the size of the input layer
   * @param inputType the type of the input layer (Dense, Sparse, SparseBinary)
   * @param inputDropout the dropout probability of the input (default 0.0).If applying it, the usual value is 0.25.
   * @param hiddenSize the size of the hidden layer
   * @param hiddenActivation the activation function of the hidden layer
   * @param hiddenDropout the dropout probability of the hidden (default 0.0).
   * @param outputSize the size of the output layer
   * @param outputActivation the activation function of the output layer
   * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
   * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
   */
  operator fun invoke(inputSize: Int,
                      inputType: LayerType.Input = LayerType.Input.Dense,
                      inputDropout: Double = 0.0,
                      hiddenSize: Int,
                      hiddenActivation: ActivationFunction?,
                      hiddenDropout: Double = 0.0,
                      outputSize: Int,
                      outputActivation: ActivationFunction?,
                      weightsInitializer: Initializer? = GlorotInitializer(),
                      biasesInitializer: Initializer? = GlorotInitializer()) = NeuralNetwork(
    LayerConfiguration(
      size = inputSize,
      inputType = inputType,
      dropout = inputDropout
    ),
    LayerConfiguration(
      size = hiddenSize,
      activationFunction = hiddenActivation,
      connectionType = LayerType.Connection.LSTM,
      dropout = hiddenDropout
    ),
    LayerConfiguration(
      size = outputSize,
      activationFunction = outputActivation,
      connectionType = LayerType.Connection.Feedforward
    ),
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer
  )
}
