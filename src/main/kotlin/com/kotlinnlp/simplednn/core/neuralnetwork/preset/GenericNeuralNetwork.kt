/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralnetwork.preset

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerInterface
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork

/**
 * A Generic [NeuralNetwork] factory.
 */
object GenericNeuralNetwork {

  /**
   * @param inputSize the size of the input layer
   * @param inputType the type of the input layer (Dense, Sparse, SparseBinary)
   * @param inputDropout the dropout probability of the input. If applying it, the usual value is 0.25.
   * @param hiddenSize the size of the hidden layers
   * @param hiddenActivation the activation function of the hidden layers
   * @param hiddenConnection the type of connection of the hidden layers
   * @param hiddenDropout the dropout probability of the hidden layers
   * @param numOfHidden the number of hidden layers (must be >= 0)
   * @param outputSize the size of the output layer
   * @param outputActivation the activation function of the output layer
   * @param weightsInitializer the initializer of the weights (zeros if null)
   * @param biasesInitializer the initializer of the biases (zeros if null)
   */
  operator fun invoke(inputSize: Int,
                      inputType: LayerType.Input,
                      inputDropout: Double,
                      hiddenSize: Int,
                      hiddenActivation: ActivationFunction?,
                      hiddenConnection: LayerType.Connection,
                      hiddenDropout: Double,
                      numOfHidden: Int,
                      outputSize: Int,
                      outputActivation: ActivationFunction?,
                      weightsInitializer: Initializer?,
                      biasesInitializer: Initializer?): NeuralNetwork {

    require(numOfHidden >= 0) { "The number of hidden layers must be >= 0." }

    val layersConfiguration = mutableListOf<LayerInterface>()

    layersConfiguration.add(LayerInterface(
      size = inputSize,
      type = inputType,
      dropout = inputDropout
    ))

    (0 until numOfHidden).forEach {
      layersConfiguration.add(LayerInterface(
        size = hiddenSize,
        activationFunction = hiddenActivation,
        connectionType = hiddenConnection,
        dropout = hiddenDropout
      ))
    }

    layersConfiguration.add(LayerInterface(
      size = outputSize,
      activationFunction = outputActivation,
      connectionType = LayerType.Connection.Feedforward
    ))

    return NeuralNetwork(
      layerConfiguration = *layersConfiguration.toTypedArray(),
      weightsInitializer = weightsInitializer,
      biasesInitializer = biasesInitializer
    )
  }
}
