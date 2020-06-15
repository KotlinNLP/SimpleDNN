/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attention.multihead

import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerInterface
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.simplednn.core.layers.models.attention.scaleddot.ScaledDotAttentionLayerParameters
import java.io.Serializable

/**
 * The parameters of the multi-head attention network.
 *
 * @property inputSize the size of the input arrays
 * @property attentionSize the size of the attention arrays
 * @property attentionOutputSize the size of the attention outputs
 * @property numOfHeads the number of self-attention heads
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
 */
class MultiHeadAttentionParameters(
  val inputSize: Int,
  val attentionSize: Int,
  val attentionOutputSize: Int,
  val numOfHeads: Int,
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
   * The size of the output arrays.
   */
  val outputSize: Int = this.inputSize

  /**
   * The parameters of the scaled-dot attention layers.
   */
  val attention: List<ScaledDotAttentionLayerParameters> = List(this.numOfHeads) {
    ScaledDotAttentionLayerParameters(
      inputSize = this.inputSize,
      attentionSize = this.attentionSize,
      outputSize = this.attentionOutputSize,
      weightsInitializer = weightsInitializer,
      biasesInitializer = biasesInitializer)
  }

  /**
   * The parameters of the output merge network.
   */
  val merge = StackedLayersParameters(
    LayerInterface(sizes = List(this.numOfHeads) { this.attentionOutputSize }),
    LayerInterface(size = this.outputSize, connectionType = LayerType.Connection.ConcatFeedforward),
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer
  )
}
