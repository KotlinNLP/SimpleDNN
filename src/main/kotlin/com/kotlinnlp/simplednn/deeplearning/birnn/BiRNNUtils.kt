/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.birnn

import com.kotlinnlp.simplednn.simplemath.concatVectorsV
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * BiRNNUtils contains functions which help to split and concatenate
 * the results of the different processors of the [BiRNNEncoder]
 */
object BiRNNUtils {

  /**
   * Split the [errorsSequence] in left-to-right and right-to-left errors
   *
   * @param errorsSequence the errors to split
   *
   * @return a (array<left-to-right>, array<right-to-left>) errors Pair
   */
  fun splitErrorsSequence(errorsSequence: List<DenseNDArray>): Pair<List<DenseNDArray>, List<DenseNDArray>> =
    errorsSequence.indices
      .map { i -> splitErrors(errorsSequence[i]) }
      .unzip()

  /**
   * Split the [errors] in left-to-right and right-to-left errors
   *
   * @param errors the errors to split
   *
   * @return a (left-to-right, right-to-left) errors Pair
   */
  fun splitErrors(errors: DenseNDArray) = Pair(
    errors.getRange(0, errors.length / 2),
    errors.getRange(errors.length / 2, errors.length))

  /**
   * Sum the left-to-right and right-to-left errors
   *
   * @params leftToRightSequenceErrors
   * @params rightToLeftSequenceErrors
   *
   * @return the sum of the left-to-right and right-to-left errors
   */
  fun sumBidirectionalErrors(leftToRightSequenceErrors: List<DenseNDArray>,
                             rightToLeftSequenceErrors: List<DenseNDArray>): List<DenseNDArray> {

    require(leftToRightSequenceErrors.size == rightToLeftSequenceErrors.size)

    return List(
      size = leftToRightSequenceErrors.size,
      init = { i ->
        leftToRightSequenceErrors[i].sum(rightToLeftSequenceErrors[leftToRightSequenceErrors.size - i - 1])
      }
    )
  }

  /**
   * Create an list of the size of [a] where each element is the concatenation of the i-th [a] and i-th [b] NDArray.
   * [b] must have the same size of [a].
   *
   * @param a a list of NDArrays
   * @param b a list of NDArrays
   *
   * @return the list of concatenated NDArrays
   */
  fun concatenate(a: List<DenseNDArray>, b: List<DenseNDArray>): List<DenseNDArray> {
    require(a.size == b.size)
    return List(size = a.size, init = { concatVectorsV(a[it], b[it]) })
  }
}
