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
import com.kotlinnlp.simplednn.core.layers.models.attention.scaleddot.ScaledDotAttentionLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.merge.concatff.ConcatFFLayerParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.preset.FeedforwardNeuralNetwork
import java.io.Serializable

/**
 * The BERT parameters.
 *
 * @param inputSize the size of the input arrays
 * @param attentionSize the size of the attention arrays
 * @param attentionOutputSize the size of the attention outputs
 * @param outputHiddenSize the number of the hidden nodes of the output feed-forward
 * @param multiHeadStack the number of scaled-dot attention layers
 * @param dropout the probability of attention dropout (default 0.0)
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
 */
class BERTParameters(
  inputSize: Int,
  attentionSize: Int,
  attentionOutputSize: Int,
  outputHiddenSize: Int,
  multiHeadStack: Int,
  val dropout: Double,
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
   * The parameters of the scaled-dot attention layers.
   */
  val attention: List<ScaledDotAttentionLayerParameters> = List(
    size = multiHeadStack,
    init = {
      ScaledDotAttentionLayerParameters(
        inputSize = inputSize,
        attentionSize = attentionSize,
        outputSize = attentionOutputSize,
        weightsInitializer = weightsInitializer)
    }
  )

  /**
   * The parameters of the merge layer of the multi-head attention outputs.
   */
  val multiHeadMerge = ConcatFFLayerParameters(
    inputsSize = List(size = multiHeadStack, init = { attentionOutputSize }),
    outputSize = inputSize,
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer)

  /**
   * The parameters of the output feed-forward network.
   */
  val outputFF = FeedforwardNeuralNetwork(
    inputSize = inputSize,
    hiddenSize = outputHiddenSize,
    hiddenActivation = ReLU(),
    outputSize = inputSize,
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
    require(multiHeadStack >= 2) { "At least 2 layers are required in the attention stack." }
  }
}
