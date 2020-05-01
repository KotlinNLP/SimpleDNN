/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.feedforward.batchnorm

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.feedforward.batchnorm.BatchNormLayer
import com.kotlinnlp.simplednn.core.layers.models.feedforward.batchnorm.BatchNormLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
object BatchNormLayerStructureUtils {

  /**
   *
   */
  fun buildLayer(): BatchNormLayer<DenseNDArray> = BatchNormLayer(
    inputArrays = listOf(
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, 0.8, -0.7, -0.5))),
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.4, -0.6, -0.2, -0.9))),
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, 0.4, 0.2, 0.8)))),
    inputType = LayerType.Input.Dense,
    params = buildParams())

  /**
   *
   */
  fun buildParams() = BatchNormLayerParameters(inputSize = 4).apply {
    g.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, 0.0, -0.3, 0.8)))
    b.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.9, 0.2, -0.9, 0.2)))
  }

  /**
   *
   */
  fun getOutputErrors1() = DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.0, -0.2, 0.4, 0.6))

  /**
   *
   */
  fun getOutputErrors2() = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.3, 0.1, 0.7, 0.9))

  /**
   *
   */
  fun getOutputErrors3() = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.3, -0.4, 0.7, -0.8))
}
