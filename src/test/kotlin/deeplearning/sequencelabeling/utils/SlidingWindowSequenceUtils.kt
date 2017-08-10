/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package deeplearning.sequencelabeling.utils

import com.kotlinnlp.simplednn.deeplearning.sequencelabeling.SlidingWindowSequence
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
object SlidingWindowSequenceUtils {

  /**
   *
   */
  fun buildSlidingWindowSequence() = SlidingWindowSequence(elements = arrayOf(
    DenseNDArrayFactory.arrayOf(doubleArrayOf(10.0, 11.0)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(20.0, 21.0)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(30.0, 31.0)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(40.0, 41.0)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(50.0, 51.0))
  ), leftContextSize = 3, rightContextSize = 3)
}
