/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.helpers

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * Utils for the calculation of the relevance.
 */
object RelevanceUtils {

  /**
   * The stabilizing term used to calculate the relevance
   */
  private const val relevanceEps: Double = 0.01

  /**
   * Calculate the relevance of the Dense array [x] respect of the calculation which produced the Dense array [y].
   *
   * @param x a [DenseNDArray]
   * @param y a [DenseNDArray] (no Sparse needed, generally little size on output)
   * @param yRelevance a [DenseNDArray], whose norm is 1.0, which indicates how much relevant are the values of [y]
   * @param contributions a matrix which contains the contributions of each value of [x] to calculate each value of [y]
   *
   * @return the relevance of [x] respect of [y]
   */
  fun calculateRelevanceOfArray(x: DenseNDArray,
                                y: DenseNDArray,
                                yRelevance: DenseNDArray,
                                contributions: DenseNDArray): DenseNDArray {

    val relevanceArray: DenseNDArray = DenseNDArrayFactory.zeros(shape = x.shape)
    val xLength: Int = x.length
    val yLength: Int = y.length

    for (i in 0 until xLength) {

      for (j in 0 until yLength) {
        val eps: Double = if (y[j] >= 0) relevanceEps else -relevanceEps
        val epsN: Double = eps / xLength

        relevanceArray[i] += yRelevance[j] * (contributions[j, i] + epsN) / (y[j] + eps)
      }
    }

    return relevanceArray
  }

  /**
   * Calculate the relevance of the Dense array [x] respect of the calculation which produced the Dense array [y].
   *
   * @param x a [DenseNDArray]
   * @param y a [DenseNDArray] (no Sparse needed, generally little size on output)
   * @param yRelevance a [DenseNDArray], whose norm is 1.0, which indicates how much relevant are the values of [y]
   * @param contributions a matrix which contains the contributions of each value of [x] to calculate each value of [y]
   *
   * @return the relevance of [x] respect of [y]
   */
  fun calculateRelevanceOfDenseArray(x: DenseNDArray,
                                     y: DenseNDArray,
                                     yRelevance: DenseNDArray,
                                     contributions: DenseNDArray): DenseNDArray {

    val relevanceArray: DenseNDArray = DenseNDArrayFactory.zeros(shape = x.shape)
    val xLength: Int = x.length
    val yLength: Int = y.length

    for (i in 0 until xLength) {

      for (j in 0 until yLength) {
        val eps: Double = if (y[j] >= 0) relevanceEps else -relevanceEps
        val epsN: Double = eps / xLength

        relevanceArray[i] += yRelevance[j] * (contributions[j, i]  + epsN) / (y[j] + eps)
      }
    }

    return relevanceArray
  }

  /**
   * @param yRelevance the relevance of [y]
   * @param y the output array
   * @param yContribute1 the first contribution to calculate [y]
   * @param yContribute2 the second contribution to calculate [y]
   * @param nPartitions the number of partitions into which [y] is divided
   *
   * @return the partition of [yRelevance] with the same ratio as [yContribute1] is in respect of [y].
   */
  fun getRelevancePartition1(yRelevance: DenseNDArray,
                             y: DenseNDArray,
                             yContribute1: DenseNDArray,
                             yContribute2: DenseNDArray,
                             nPartitions: Int = 2): DenseNDArray {

    val eps: DenseNDArray = yContribute2.nonZeroSign().assignProd(relevanceEps) // the same factor (yContribute2)
    // is needed to calculate eps either for the first partition then the second one

    // partition factor = (yContribute1 + eps / n) / (yContribute1 + yContribute2 + eps) [eps avoids divisions by zero]
    return yRelevance.prod(yContribute1.sum(eps.div(nPartitions.toDouble()))).assignDiv(y.sum(eps))
  }

  /**
   * @param yRelevance the relevance of [y]
   * @param y the output array
   * @param yContribute2 the second contribution to calculate [y]
   * @param nPartitions the number of partitions into which [y] is divided
   *
   * @return the partition of [yRelevance] with the same ratio as [yContribute2] is in respect of [y].
   */
  fun getRelevancePartition2(yRelevance: DenseNDArray,
                             y: DenseNDArray,
                             yContribute2: DenseNDArray,
                             nPartitions: Int = 2): DenseNDArray {

    val eps: DenseNDArray = yContribute2.nonZeroSign().assignProd(relevanceEps)

    // partition factor = (yContribute2 + eps / n) / (yInput + yContribute2 + eps) [eps avoids divisions by zero]
    return yRelevance.prod(yContribute2.sum(eps.div(nPartitions.toDouble()))).assignDiv(y.sum(eps))
  }
}
