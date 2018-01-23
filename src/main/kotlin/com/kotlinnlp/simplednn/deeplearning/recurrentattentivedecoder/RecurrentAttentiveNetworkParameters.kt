/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.recurrentattentivedecoder

import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.optimizer.IterableParams
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.AttentionNetworkParameters

/**
 * The parameters of the [RecurrentAttentiveNetwork].
 */
class RecurrentAttentiveNetworkParameters(
  private val attentionParams: AttentionNetworkParameters,
  private val transformParams: FeedforwardLayerParameters,
  private val recurrentContextParams: NetworkParameters,
  private val outputParams: NetworkParameters
) : IterableParams<RecurrentAttentiveNetworkParameters>() {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }
  /**
   * The list of all parameters.
   */
  override val paramsList: Array<UpdatableArray<*>> =
    this.attentionParams.paramsList +
      this.transformParams.paramsList +
      this.recurrentContextParams.paramsList +
      this.outputParams.paramsList


  /**
   * @return a new [RecurrentAttentiveNetworkParameters] containing a copy of all values of this
   */
  override fun copy() = RecurrentAttentiveNetworkParameters(
    attentionParams = this.attentionParams.copy(),
    transformParams = this.transformParams.copy(),
    recurrentContextParams = this.recurrentContextParams.copy(),
    outputParams = this.outputParams.copy())
}
