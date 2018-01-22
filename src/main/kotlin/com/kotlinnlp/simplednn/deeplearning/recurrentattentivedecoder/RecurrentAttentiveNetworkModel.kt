/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.recurrentattentivedecoder

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.layers.LayerConfiguration
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.AttentionNetworkParameters

/**
 * The model of the [RecurrentAttentiveNetwork].
 *
 * @property inputSize the size of the input items of a sequence
 * @property attentionSize the size of the attention arrays
 * @property recurrentContextSize the size of the recurrent context
 * @property outputSize the output size of the final output network
 * @property labelSize the size of the label resulting form a final prediction
 * @param contextActivation the activation of the recurrent network that encode the context
 * @param contextRecurrenceType the recurrent layer type (e.g. LSTM, GRU, RAN, ...)
 * @param outputActivationFunction the activation function of the final output network
 */
class RecurrentAttentiveNetworkModel(
  val inputSize: Int,
  val attentionSize: Int,
  val recurrentContextSize: Int,
  val outputSize: Int,
  val labelSize: Int,
  contextActivation: ActivationFunction,
  contextRecurrenceType: LayerType.Connection,
  outputActivationFunction: ActivationFunction
){

  /**
   * The parameters of the attention network.
   */
  val attentionParams = AttentionNetworkParameters(
    inputSize = this.inputSize,
    attentionSize = this.attentionSize)
  
  /**
   * The parameters of the transform layers used to create the attention arrays of the [attentionParams].
   */
  val transformParams = FeedforwardLayerParameters(
    inputSize = this.inputSize + this.recurrentContextSize,
    outputSize = this.attentionSize)

  /**
   * The RNN used to encode the state.
   */
  val contextNetwork = NeuralNetwork(
    LayerConfiguration(
      size = this.attentionParams.outputSize + labelSize,
      inputType = LayerType.Input.Dense
    ),
    LayerConfiguration(
      size = this.recurrentContextSize,
      activationFunction = contextActivation,
      connectionType = contextRecurrenceType
    )
  )

  /**
   * The output network.
   */
  val outputNetwork = NeuralNetwork(
    LayerConfiguration(
      size = this.attentionParams.outputSize + this.recurrentContextSize,
      inputType = LayerType.Input.Dense,
      dropout = 0.0),
    LayerConfiguration(
      size = this.outputSize,
      connectionType = LayerType.Connection.Feedforward,
      activationFunction = outputActivationFunction,
      meProp = false)
  )
}
