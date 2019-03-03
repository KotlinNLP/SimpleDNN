/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.simple

import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.getInputRelevance
import com.kotlinnlp.simplednn.core.layers.helpers.RelevanceHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * The helper which calculates the relevance of the input of a [layer] respect of its output.
 *
 * @property layer the [FeedforwardLayer] in which to calculate the input relevance
 */
class FeedforwardRelevanceHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: FeedforwardLayer<InputNDArrayType>
) : RelevanceHelper<InputNDArrayType>(layer) {

  /**
   * @param layerContributions the contributions saved during the last forward
   *
   * @return the relevance of the input respect of the output
   */
  override fun getInputRelevance(layerContributions: LayerParameters<*>): NDArray<*> {
    layerContributions as FeedforwardLayerParameters

    return this.layer.outputArray.getInputRelevance(
      x = this.layer.inputArray.values,
      cw = layerContributions.unit.weights.values)
  }
}
