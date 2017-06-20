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
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which calculates the relevance of the input of a [layer] respect of its output.
 *
 * @property layer the [RecurrentLayerStructure] in which to calculate the input relevance
 */
abstract class RecurrentRelevanceHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: RecurrentLayerStructure<InputNDArrayType>
) : RelevanceHelper<InputNDArrayType>(layer) {

  /**
   * Calculate the relevance of the output in the previous state respect of the current one.
   *
   * @param layerContributions the structure in which to save the contributions during the calculations
   */
  abstract fun calculateRecurrentRelevance(layerContributions: LayerParameters)

  /**
   * Get the partition of the output relevance respect of the input.
   *
   * @param y the output array
   * @param yInput the contribution to calculate the output array coming from the input
   * @param yRec the contribution to calculate the output array coming from the recursion
   *
   * @return the partition of the output relevance respect of the input
   */
  protected fun getInputRelevancePartition(y: DenseNDArray, yInput: DenseNDArray, yRec: DenseNDArray): DenseNDArray {

    val yRelevance: DenseNDArray = this.layer.outputArray.relevance as DenseNDArray
    val eps: DenseNDArray = yRec.nonZeroSign().assignProd(this.relevanceEps) // the same factor (yRec) is needed
    // to calculate eps either for the input partition then the recurrent one

    // partition factor = (yInput + eps / 2) / (yInput + yRec + eps) [eps avoids divisions by zero]
    return yRelevance.prod(yInput.sum(eps.div(2.0))).assignDiv(y.sum(eps))
  }
}
