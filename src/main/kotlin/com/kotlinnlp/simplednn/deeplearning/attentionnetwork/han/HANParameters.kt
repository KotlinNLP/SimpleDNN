/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attentionnetwork.han

import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.optimizer.IterableParams
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.AttentionNetworkParameters
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNParameters

/**
 * The parameters of the Hierarchical Attention Networks.
 *
 * @property biRNNs an array containing the parameters of the BiRNNs of the HAN.
 * @property attentionNetworks an array containing the parameters of the AttentionNetworks of the HAN
 * @property outputNetwork the parameters of the output Feedforward network
 */
class HANParameters(
  val biRNNs: Array<BiRNNParameters>,
  val attentionNetworks: Array<AttentionNetworkParameters>,
  val outputNetwork: NetworkParameters
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
  override val paramsList: Array<UpdatableArray<*>> = this.buildParamsList()

  /**
   * @return a new [HANParameters] containing a copy of all values of this
   */
  override fun copy() = HANParameters(
    biRNNs = Array(size = this.biRNNs.size, init = { i -> this.biRNNs[i].copy() }),
    attentionNetworks = Array(size = this.attentionNetworks.size, init = { i -> this.attentionNetworks[i].copy() }),
    outputNetwork = this.outputNetwork.copy()
  )

  /**
   * @return a new [HANParameters] containing a copy of all values of this
   */
  private fun buildParamsList(): Array<UpdatableArray<*>> {

    val params = arrayListOf<UpdatableArray<*>>()

    this.biRNNs.forEach { birnnParams -> birnnParams.forEach { params.add(it) } }
    this.attentionNetworks.forEach { attentionNetworkParams -> attentionNetworkParams.forEach { params.add(it) } }
    outputNetwork.forEach { params.add(it) }

    return params.toTypedArray()
  }
}
