/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.transformers

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ReLU
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.neuralnetwork.preset.FeedforwardNeuralNetwork
import com.kotlinnlp.simplednn.deeplearning.attention.multihead.MultiHeadAttentionParameters
import java.io.Serializable

/**
 * The BERT parameters.
 *
 * @property inputSize the size of the input arrays
 * @property attentionSize the size of the attention arrays
 * @property attentionOutputSize the size of the attention outputs
 * @property outputHiddenSize the number of the hidden nodes of the output feed-forward
 * @property multiHeadStack the number of scaled-dot attention layers
 * @param dropout the probability of attention dropout (default 0.0)
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
 */
class BERTParameters(
  val inputSize: Int,
  val attentionSize: Int,
  val attentionOutputSize: Int,
  val outputHiddenSize: Int,
  val multiHeadStack: Int,
  dropout: Double,
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = GlorotInitializer()
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The parameters of the multi-head scaled-dot attention network.
   */
  val attention = MultiHeadAttentionParameters(
    inputSize = this.inputSize,
    attentionSize = this.attentionSize,
    attentionOutputSize = this.attentionOutputSize,
    numOfLayers = this.multiHeadStack,
    dropout = dropout,
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer)

  /**
   * The parameters of the output feed-forward network.
   */
  val outputFF = FeedforwardNeuralNetwork(
    inputSize = this.inputSize,
    hiddenSize = this.outputHiddenSize,
    hiddenActivation = ReLU(),
    outputSize = this.inputSize,
    outputActivation = null,
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer)

  /**
   * The updatable normalization scalar parameter.
   */
  var normScalarParam = ParamsArray(doubleArrayOf(0.0))

  /**
   * The scalar parameter used to normalize the merged vectors.
   */
  val normScalar: Double get() = this.normScalarParam.values[0]

  /**
   * Check requirements.
   */
  init {
    require(this.multiHeadStack >= 2) { "At least 2 layers are required in the attention stack." }
  }
}
