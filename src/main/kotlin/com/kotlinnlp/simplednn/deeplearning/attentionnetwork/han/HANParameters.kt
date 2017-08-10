/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attentionnetwork.han

import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.AttentionNetworkParameters
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNParameters

/**
 * The parameters of the Hierarchical Attention Networks.
 *
 * @property biRNNs an array containing the parameters of the BiRNNs of the HAN.
 * @property attentionNetworks an array containing the parameters of the AttentionNetworks of the HAN
 * @property outputNetwork the parameters of the output Feedforward network
 */
data class HANParameters(
  val biRNNs: ArrayList<BiRNNParameters>,
  val attentionNetworks: ArrayList<AttentionNetworkParameters>,
  val outputNetwork: NetworkParameters
)
