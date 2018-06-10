/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.helpers

import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.LayerStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * The helper which calculates the relevance of the input of a [layer] respect of its output.
 *
 * @property layer the [LayerStructure] in which to calculate the input relevance
 */
abstract class RelevanceHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  protected open val layer: LayerStructure<InputNDArrayType>
) {

  /**
   * Calculate the relevance of the input respect of the output and assign it to the input array.
   *
   * @param layerContributions the contributions saved during the last forward
   */
  fun setInputRelevance(layerContributions: LayerParameters<*>) {
    this.layer.inputArray.assignRelevance(relevance = this.getInputRelevance(layerContributions = layerContributions))
  }

  /**
   * Calculate the relevance of the input respect of the output and add it to the relevance of the input array
   * previously set.
   *
   * @param layerContributions the contributions saved during the last forward
   */
  fun addInputRelevance(layerContributions: LayerParameters<*>) {
    this.layer.inputArray.relevance.assignSum(this.getInputRelevance(layerContributions = layerContributions))
  }

  /**
   * Calculate the relevance of the input respect of the output.
   *
   * @param layerContributions the contributions saved during the last forward
   *
   * @return the relevance of the input in respect of the output
   */
  protected abstract fun getInputRelevance(layerContributions: LayerParameters<*>): NDArray<*>
}
