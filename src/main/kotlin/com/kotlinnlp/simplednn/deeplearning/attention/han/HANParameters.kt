/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attention.han

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.simplednn.core.optimizer.IterableParams
import com.kotlinnlp.simplednn.deeplearning.attention.attentionnetwork.AttentionNetworkParameters
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNParameters

/**
 * The parameters of the Hierarchical Attention Networks.
 *
 * @property biRNNs a list containing the parameters of the BiRNNs of the HAN.
 * @property attentionNetworks a list containing the parameters of the AttentionNetworks of the HAN
 * @property outputStackedLayers the parameters of the output Feedforward network
 */
class HANParameters(
  val biRNNs: List<BiRNNParameters>,
  val attentionNetworks: List<AttentionNetworkParameters>,
  val outputStackedLayers: StackedLayersParameters
) : IterableParams<HANParameters>() {

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
  override val paramsList: List<ParamsArray> =
    this.biRNNs.flatMap { it.paramsList } +
      this.attentionNetworks.flatMap { it.paramsList } +
      this.outputStackedLayers.paramsList

  /**
   * @return a new [HANParameters] containing a copy of all values of this
   */
  override fun copy() = HANParameters(
    biRNNs = this.biRNNs.map { it.copy() },
    attentionNetworks = this.attentionNetworks.map { it.copy() },
    outputStackedLayers = this.outputStackedLayers.copy()
  )
}
