/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.mergelayers.concat

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.mergelayers.concat.ConcatLayerParameters
import com.kotlinnlp.simplednn.core.mergelayers.concat.ConcatLayerStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
object ConcatLayerUtils {

  /**
   *
   */
  fun buildLayer(): ConcatLayerStructure = ConcatLayerStructure(
    inputArrays = listOf(
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.9, 0.9, 0.6, 0.1))),
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.5))),
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.7, -0.7, 0.8)))
    ),
    outputArray = AugmentedArray(size = 9),
    params = ConcatLayerParameters(inputsSize = listOf(4, 2, 3))
  )

  /**
   *
   */
  fun getOutputErrors() = DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.0, -0.2, 0.4, -0.2, 0.0, -0.7, 0.2, -0.1, -0.7))
}
