/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.merge.concat

import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.helpers.RelevanceHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * The helper which calculates the relevance of the input of a [ConcatLayer] respect of its output.
 *
 * @property layer the layer in which to calculate the input relevance
 */
class ConcatRelevanceHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: ConcatLayer<InputNDArrayType>
) : RelevanceHelper<InputNDArrayType>(layer) {

  /**
   * Not available for the Concat layer.
   *
   * @param layerContributions the contributions saved during the last forward
   *
   * @return the relevance of the input respect of the output
   */
  override fun getInputRelevance(layerContributions: LayerParameters<*>): NDArray<*> {
    throw NotImplementedError("Relevance not available for the Concat layer.")
  }
}
