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
   * @param yRelevance the relevance of [y]
   * @param y the output array
   * @param yContribute1 the first contribution to calculate [y]
   * @param yContribute2 the second contribution to calculate [y]
   *
   * @return the partition of [yRelevance] with the same ratio as [yContribute1] is in respect of [y].
   */
  protected fun getRelevancePartition1(yRelevance: DenseNDArray,
                                       y: DenseNDArray,
                                       yContribute1: DenseNDArray,
                                       yContribute2: DenseNDArray): DenseNDArray {

    val eps: DenseNDArray = yContribute2.nonZeroSign().assignProd(this.relevanceEps) // the same factor (yContribute2)
    // is needed to calculate eps either for the first partition then the second one

    // partition factor = (yContribute1 + eps / 2) / (yContribute1 + yContribute2 + eps) [eps avoids divisions by zero]
    return yRelevance.prod(yContribute1.sum(eps.div(2.0))).assignDiv(y.sum(eps))
  }

  /**
   * Get the partition of the output relevance respect of the output in the previous state.
   *
   * @param yRelevance the relevance of [y]
   * @param y the output array
   * @param yContribute2 the second contribution to calculate [y]
   *
   * @return the partition of [yRelevance] with the same ratio as [yContribute2] is in respect of [y].
   */
  protected fun getRelevancePartition2(yRelevance: DenseNDArray,
                                       y: DenseNDArray,
                                       yContribute2: DenseNDArray): DenseNDArray {

    val eps: DenseNDArray = yContribute2.nonZeroSign().assignProd(this.relevanceEps)

    // partition factor = (yContribute2 + eps / 2) / (yInput + yContribute2 + eps) [eps avoids divisions by zero]
    return yRelevance.prod(yContribute2.sum(eps.div(2.0))).assignDiv(y.sum(eps))
  }
}
