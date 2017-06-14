/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent

import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.RelevanceHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * The helper which calculates the relevance of the input of a [layer] respect of its output.
 *
 * @property layer the [RecurrentLayerStructure] in which to calculate the input relevance
 */
abstract class RecurrentRelevanceHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: RecurrentLayerStructure<InputNDArrayType>
) : RelevanceHelper<InputNDArrayType>(layer) {

  /**
   * Calculate the relevance of the input respect of the output and add it to the relevance of the input array
   * previously set.
   *
   * @param paramsContributes the contributes of the parameters during the last forward
   */
  abstract fun addInputRelevance(paramsContributes: LayerParameters)

  /**
   * Calculate the relevance of the output in the previous state respect of the current one.
   *
   * @param paramsContributes the contributes of the parameters during the last forward
   */
  abstract fun calculateRecurrentRelevance(paramsContributes: LayerParameters)
}
