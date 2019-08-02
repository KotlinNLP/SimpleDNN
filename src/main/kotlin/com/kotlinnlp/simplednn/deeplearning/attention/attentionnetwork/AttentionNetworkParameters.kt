/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attention.attentionnetwork

import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerParametersFactory
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.attention.attentionmechanism.AttentionMechanismLayerParameters

/**
 * The parameters of the Attention Network.
 *
 * @property inputSize the size of the input arrays
 * @property attentionSize the size of the attention arrays
 * @property sparseInput whether the input arrays are sparse
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
 */
class AttentionNetworkParameters(
  val inputSize: Int,
  val attentionSize: Int,
  val sparseInput: Boolean = false,
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = GlorotInitializer()
) {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The size of the output array.
   */
  val outputSize: Int = this.inputSize

  /**
   * The parameters of the transform layer.
   */
  val transformParams: FeedforwardLayerParameters = LayerParametersFactory(
    inputsSize = listOf(this.inputSize),
    outputSize = this.attentionSize,
    connectionType = LayerType.Connection.Feedforward,
    sparseInput = this.sparseInput,
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer
  ) as FeedforwardLayerParameters

  /**
   * The parameters of the attention layer.
   */
  val attentionParams = AttentionMechanismLayerParameters(
    inputSize = this.attentionSize,
    weightsInitializer = weightsInitializer)
}
