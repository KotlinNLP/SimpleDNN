/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.merge.sum

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.models.merge.sum.SumLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.merge.sum.SumLayer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
object SumLayerUtils {

  /**
   *
   */
  fun buildLayer(): SumLayer<DenseNDArray> = SumLayer(
    inputArrays = listOf(
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.9, 0.9, 0.6))),
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.5, -0.5))),
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.7, -0.7, 0.8))),
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.5, -0.4, -0.8)))
    ),
    outputArray = AugmentedArray(size = 3),
    params = SumLayerParameters(inputSize = 3, nInputs = 4)
  )

  /**
   *
   */
  fun getOutputErrors() = DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.0, -0.2, 0.4))
}
