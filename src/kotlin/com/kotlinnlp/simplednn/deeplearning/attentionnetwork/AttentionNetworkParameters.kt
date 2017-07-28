/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attentionnetwork

import com.kotlinnlp.simplednn.core.layers.LayerParametersFactory
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.attentionlayer.AttentionLayerParameters

/**
 * The parameters of the Attention Network.
 */
data class AttentionNetworkParameters(
  val inputSize: Int,
  val attentionSize: Int,
  val sparseInput: Boolean = false
) {

  /**
   * The size of the output array.
   */
  val outputSize: Int = this.inputSize

  /**
   * The parameters of the transform layer.
   */
  val transformParams: FeedforwardLayerParameters = LayerParametersFactory(
    inputSize = this.inputSize,
    outputSize = this.attentionSize,
    connectionType = LayerType.Connection.Feedforward,
    sparseInput = this.sparseInput) as FeedforwardLayerParameters

  /**
   * The parameters of the attention layer.
   */
  val attentionParams = AttentionLayerParameters(attentionSize = this.attentionSize)
}
