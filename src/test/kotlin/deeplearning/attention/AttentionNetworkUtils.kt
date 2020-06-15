/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package deeplearning.attention

import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.deeplearning.attention.attentionnetwork.AttentionNetworkParameters
import com.kotlinnlp.simplednn.deeplearning.attention.attentionnetwork.AttentionNetwork
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import core.attention.AttentionLayerUtils

/**
 *
 */
internal object AttentionNetworkUtils {

  /**
   *
   */
  fun buildNetwork() = AttentionNetwork<DenseNDArray>(
    inputType = LayerType.Input.Dense,
    model = buildAttentionNetworkParams(),
    propagateToInput = true)

  /**
   *
   */
  private fun buildAttentionNetworkParams() = AttentionNetworkParameters(inputSize = 4, attentionSize = 2).apply {

    val transformParams = AttentionLayerUtils.buildTransformLayerParams()
    val attentionParams = AttentionLayerUtils.buildAttentionParams()

    transform.getLayerParams<FeedforwardLayerParameters>(0).unit.weights.values.assignValues(
      transformParams.unit.weights.values)

    transform.getLayerParams<FeedforwardLayerParameters>(0).unit.biases.values.assignValues(
      transformParams.unit.biases.values)

    attention.contextVector.values.assignValues(attentionParams.contextVector.values)
  }
}
