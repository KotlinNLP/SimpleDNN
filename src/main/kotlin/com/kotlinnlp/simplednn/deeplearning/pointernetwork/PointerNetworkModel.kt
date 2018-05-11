/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.pointernetwork

import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.mergelayers.affine.AffineLayerParameters
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.attentionmechanism.AttentionParameters
import java.io.Serializable


/**
 * The model of the [PointerNetwork].
 *
 * @property inputSize the size of the elements of the input sequence
 * @property vectorSize the size of the vector that modulates a content-based attention mechanism over the input sequence
 * @property attentionSize the size of the attention vectors
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: null)
 */
class PointerNetworkModel(
  val inputSize: Int,
  val vectorSize: Int,
  val attentionSize: Int,
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = null) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The parameters used to create the attention arrays of the [attentionParams].
   */
  val transformParams = AffineLayerParameters(
    inputSize1 = this.inputSize,
    inputSize2 = this.vectorSize,
    outputSize = this.attentionSize,
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer)

  /**
   * The parameters of the attention mechanism.
   */
  val attentionParams = AttentionParameters(
    attentionSize = this.transformParams.outputSize,
    initializer = weightsInitializer)

  /**
   * The structure containing all the parameters of this model.
   */
  val params = PointerNetworkParameters(
    transformParams = this.transformParams,
    attentionParams = this.attentionParams)
}
