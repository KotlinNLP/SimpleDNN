/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attentiverecurrentnetwork

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.layers.LayerConfiguration
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.deeplearning.attention.attentionnetwork.AttentionNetworkParameters
import java.io.Serializable

/**
 * The model of the [AttentiveRecurrentNetwork].
 *
 * @property inputSize the size of the input sequence arrays
 * @property recurrentContextSize the size of the recurrent context (the output of the recurrent context network)
 * @property contextLabelSize the size of the context label vector (as input of the recurrent context network)
 * @property outputSize the output size
 * @param contextActivation the activation of the recurrent network that encodes the context
 * @param contextRecurrenceType the recurrent layer type (e.g. LSTM, GRU, RAN, ...)
 * @param outputActivationFunction the activation function of the final output network
 */
class AttentiveRecurrentNetworkModel(
  val inputSize: Int,
  val attentionSize: Int,
  val recurrentContextSize: Int,
  val contextLabelSize: Int,
  val outputSize: Int,
  contextActivation: ActivationFunction,
  contextRecurrenceType: LayerType.Connection,
  outputActivationFunction: ActivationFunction
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

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
   * The RNN used to merge the Attention Network output together with the context vector.
   */
  val recurrentContextNetwork = NeuralNetwork(
    LayerConfiguration(
      size = this.attentionParams.outputSize + this.contextLabelSize,
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

  /**
   * The structure containing all the parameters of this model.
   */
  val params = AttentiveRecurrentNetworkParameters(
    attentionParams = this.attentionParams,
    transformParams = this.transformParams,
    recurrentContextParams = this.recurrentContextNetwork.model,
    outputParams = this.outputNetwork.model)
}
