/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralnetwork.preset

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.layers.LayerConfiguration
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork

/**
 *
 */
object LSTMNeuralNetwork {

  operator fun invoke(inputSize: Int,
                      hiddenSize: Int,
                      hiddenActivation: ActivationFunction?,
                      outputSize: Int,
                      outputActivation: ActivationFunction?,
                      inputType: LayerType.Input = LayerType.Input.Dense,
                      dropout: Double = 0.0) = NeuralNetwork(
    LayerConfiguration(
      size = inputSize,
      inputType = inputType
    ),
    LayerConfiguration(
      size = hiddenSize,
      activationFunction = hiddenActivation,
      connectionType = LayerType.Connection.LSTM,
      dropout = dropout
    ),
    LayerConfiguration(
      size = outputSize,
      activationFunction = outputActivation,
      connectionType = LayerType.Connection.Feedforward)
  )
}
