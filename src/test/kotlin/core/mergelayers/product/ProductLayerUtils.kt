/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.mergelayers.product

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.mergelayers.product.ProductLayerParameters
import com.kotlinnlp.simplednn.core.mergelayers.product.ProductLayerStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
object ProductLayerUtils {

  /**
   *
   */
  fun buildLayer4(): ProductLayerStructure = ProductLayerStructure(
    inputArrays = listOf(
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.9, 0.9, 0.6))),
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.5, -0.5))),
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.7, -0.7, 0.8))),
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.5, -0.4, -0.8)))
    ),
    outputArray = AugmentedArray(size = 3),
    params = ProductLayerParameters(inputSize = 3, nInputs = 4)
  )

  /**
   *
   */
  fun buildLayer5(): ProductLayerStructure = ProductLayerStructure(
    inputArrays = listOf(
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.9, 0.9, 0.6))),
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.5, -0.5))),
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.7, -0.7, 0.8))),
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.5, -0.4, -0.8))),
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.5, -0.9, -0.5)))
    ),
    outputArray = AugmentedArray(size = 3),
    params = ProductLayerParameters(inputSize = 3, nInputs = 5)
  )

  /**
   *
   */
  fun getOutputErrors4() = DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.0, -0.2, 0.4))

  /**
   *
   */
  fun getOutputErrors5() = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.7, -0.5, -0.8))
}
