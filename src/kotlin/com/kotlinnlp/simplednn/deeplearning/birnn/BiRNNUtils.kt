/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.birnn

import com.kotlinnlp.simplednn.simplemath.NDArray
import com.kotlinnlp.simplednn.simplemath.concatVectorsV

/**
 * BiRNNUtils contains functions which help to split and concatenate
 * the results of the different processors of the [BiRNNEncoder]
 */
object BiRNNUtils {

  /**
   * Split the [array] in left-to-right and right-to-left errors
   *
   * @param array errors to split
   *
   * @return a (array<left-to-right>, array<right-to-left>) errors Pair
   */
  fun splitErrorsSequence(array: Array<NDArray>): Pair<Array<NDArray>, Array<NDArray>> {

    val leftToRightOutputErrors = arrayOfNulls<NDArray>(array.size)
    val rightToLeftOutputErrors = arrayOfNulls<NDArray>(array.size)

    array.indices.forEach { i -> val (a, b) = this.splitErrors(array[i])
      leftToRightOutputErrors[i] = a
      rightToLeftOutputErrors[i] = b
    }

    return Pair(leftToRightOutputErrors.requireNoNulls(), rightToLeftOutputErrors.requireNoNulls())
  }

  /**
   * Split the [array] in left-to-right and right-to-left errors
   *
   * @param array errors to split
   *
   * @return a (left-to-right, right-to-left) errors Pair
   */
  fun splitErrors(array: NDArray) = Pair(
    array.getRange(0, array.length / 2),
    array.getRange(array.length / 2, array.length))

  /**
   * Sum the left-to-right and right-to-left errors
   *
   * @params leftToRightInputErrors
   * @params rightToLeftInputErrors
   *
   * @return the sum of the left-to-right and right-to-left errors
   */
  fun sumBidirectionalErrors(leftToRightInputErrors: Array<NDArray>, rightToLeftInputErrors: Array<NDArray>): Array<NDArray> {
    require(leftToRightInputErrors.size == rightToLeftInputErrors.size)

    return Array(size = leftToRightInputErrors.size, init = {
      leftToRightInputErrors[it].sum(rightToLeftInputErrors[leftToRightInputErrors.size - it - 1])
    })
  }

  /**
   * Create an an array of the size of [a] where each element
   * is the concatenation of the i-th [a] and i-th [b] NDArray
   *
   * @param a array of NDArray
   * @param b array of NDArray
   *          WARNING: [b] must have the same size of [a]
   *
   * @return the array resulting from the concatenation
   */
  fun concatenate(a: Array<NDArray>, b: Array<NDArray>): Array<NDArray> {
    require(a.size == b.size)
    return Array(size = a.size, init = { concatVectorsV(a[it], b[it]) })
  }
}
