/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attention.han

import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.simplednn.deeplearning.attention.attentionnetwork.AttentionNetworkParameters
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNParameters
import java.io.Serializable

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
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }
}
