/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.birnn

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * BiRNNUtils contains functions which help to split and concatenate
 * the results of the different processors of the [BiRNNEncoder]
 */
object BiRNNUtils {

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
}
