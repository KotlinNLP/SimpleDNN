/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.helpers

import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.Layer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which calculates the relevance of the input of a [layer] respect of its output.
 *
 * @property layer the [Layer] in which to calculate the input relevance
 */
internal abstract class RelevanceHelper(protected open val layer: Layer<DenseNDArray>) {

  /**
   * Calculate the relevance of the input respect to the output and assign it to the input array.
   *
   * @param contributions the contributions saved during the last forward
   */
  fun setInputRelevance(contributions: LayerParameters) {
    this.layer.inputArray.assignRelevance(relevance = this.getInputRelevance(contributions = contributions))
  }

  /**
   * Calculate the relevance of the input respect to the output and add it to the relevance of the input array
   * previously set.
   *
   * @param contributions the contributions saved during the last forward
   */
  fun addInputRelevance(contributions: LayerParameters) {
    this.layer.inputArray.relevance.assignSum(this.getInputRelevance(contributions = contributions))
  }

  /**
   * @param contributions the contributions saved during the last forward
   *
   * @return the relevance of the input respect to the output
   */
  protected abstract fun getInputRelevance(contributions: LayerParameters): DenseNDArray
}
