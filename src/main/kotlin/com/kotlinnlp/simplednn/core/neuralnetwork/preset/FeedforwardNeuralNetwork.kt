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
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters

/**
 * The Feed-forward neural network factory.
 */
object FeedforwardNeuralNetwork {

  /**
   * @param inputSize the size of the input layer
   * @param inputType the type of the input layer (Dense -default-, Sparse, SparseBinary)
   * @param hiddenSize the size of the hidden layers
   * @param hiddenActivation the activation function of the hidden layers
   * @param numOfHidden the number of hidden layers (must be >= 0, default 1)
   * @param outputSize the size of the output layer
   * @param outputActivation the activation function of the output layer
   * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
   * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
   */
  operator fun invoke(inputSize: Int,
                      inputType: LayerType.Input = LayerType.Input.Dense,
                      hiddenSize: Int,
                      hiddenActivation: ActivationFunction?,
                      numOfHidden: Int = 1,
                      outputSize: Int,
                      outputActivation: ActivationFunction?,
                      weightsInitializer: Initializer? = GlorotInitializer(),
                      biasesInitializer: Initializer? = GlorotInitializer()): StackedLayersParameters =
    GenericNeuralNetwork(
      inputSize = inputSize,
      inputType = inputType,
      hiddenSize = hiddenSize,
      hiddenActivation = hiddenActivation,
      hiddenConnection = LayerType.Connection.Feedforward,
      numOfHidden = numOfHidden,
      outputSize = outputSize,
      outputActivation = outputActivation,
      weightsInitializer = weightsInitializer,
      biasesInitializer = biasesInitializer
    )
}
