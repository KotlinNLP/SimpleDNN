/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers

import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * The helper which calculates the relevance of the input of a [layer] respect of its output.
 */
interface RelevanceHelper<InputNDArrayType : NDArray<InputNDArrayType>> {

  /**
   * The [LayerStructure] in which to calculate the input relevance.
   */
  val layer: LayerStructure<InputNDArrayType>

  /**
   * Calculate the relevance of the input.
   *
   * @param paramsContributes the contributes of the parameters during the last forward
   */
  fun calculateRelevance(paramsContributes: LayerParameters)
}
