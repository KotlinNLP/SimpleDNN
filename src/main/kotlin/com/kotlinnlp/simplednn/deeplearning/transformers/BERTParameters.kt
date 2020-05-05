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
import com.kotlinnlp.simplednn.core.layers.LayerInterface
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
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
 * @property numOfHeads the number of self-attention heads
 * @param dropout the probability of attention dropout (default 0.0)
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
 */
class BERTParameters(
  val inputSize: Int,
  val attentionSize: Int,
  val attentionOutputSize: Int,
  val outputHiddenSize: Int,
  val numOfHeads: Int,
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
    numOfHeads = this.numOfHeads,
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
   * The parameters of the multi-head norm layer.
   */
  val multiHeadNorm = StackedLayersParameters(
    LayerInterface(size = this.inputSize),
    LayerInterface(size = this.inputSize, connectionType = LayerType.Connection.Norm),
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer)

  /**
   * The parameters of the output norm layer.
   */
  val outputNorm = StackedLayersParameters(
    LayerInterface(size = this.inputSize),
    LayerInterface(size = this.inputSize, connectionType = LayerType.Connection.Norm),
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer)

  /**
   * Check requirements.
   */
  init {
    require(this.numOfHeads >= 2) { "At least 2 heads are required in the attention stack." }
  }
}
