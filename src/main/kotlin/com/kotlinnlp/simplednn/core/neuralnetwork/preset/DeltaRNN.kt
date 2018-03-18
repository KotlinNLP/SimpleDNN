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
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork

/**
 * The Delta Recurrent Neural Network factory.
 */
object DeltaRNN {

  /**
   * @param inputSize the size of the input layer
   * @param inputType the type of the input layer (Dense -default-, Sparse, SparseBinary)
   * @param inputDropout the dropout probability of the input (default 0.0). If applying it, the usual value is 0.25.
   * @param hiddenSize the size of the hidden layers
   * @param hiddenActivation the activation function of the hidden layers
   * @param hiddenDropout the dropout probability of the hidden layers (default 0.0)
   * @param numOfHidden the number of hidden layers (must be >= 0, default 1)
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
                      numOfHidden: Int = 1,
                      outputSize: Int,
                      outputActivation: ActivationFunction?,
                      weightsInitializer: Initializer? = GlorotInitializer(),
                      biasesInitializer: Initializer? = GlorotInitializer()): NeuralNetwork =
    GenericNeuralNetwork(
      inputSize = inputSize,
      inputType = inputType,
      inputDropout = inputDropout,
      hiddenSize = hiddenSize,
      hiddenActivation = hiddenActivation,
      hiddenDropout = hiddenDropout,
      hiddenMeProp = false,
      hiddenConnection = LayerType.Connection.DeltaRNN,
      numOfHidden = numOfHidden,
      outputSize = outputSize,
      outputActivation = outputActivation,
      outputMeProp = false,
      weightsInitializer = weightsInitializer,
      biasesInitializer = biasesInitializer
    )
}
