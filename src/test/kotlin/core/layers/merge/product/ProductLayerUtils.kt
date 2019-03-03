/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.merge.product

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.models.merge.product.ProductLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.merge.product.ProductLayer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
object ProductLayerUtils {

  /**
   *
   */
  fun buildLayer(): ProductLayer<DenseNDArray> = ProductLayer(
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
  fun getOutputErrors() = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.7, -0.5, -0.8))
}
