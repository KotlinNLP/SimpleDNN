/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.feedforward.normalization

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.feedforward.normalization.NormalizationLayer
import com.kotlinnlp.simplednn.core.layers.models.feedforward.normalization.NormalizationLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

object NormalizationLayerStructureUtils {

  /**
   *
   */
  fun buildLayer(): NormalizationLayer<DenseNDArray> = NormalizationLayer(
      inputArrays = listOf(
          AugmentedArray(values = buildarray1()),
          AugmentedArray(values = buildarray2()),
          AugmentedArray(values = buildarray3())),
      inputSize = 4,
      inputType = LayerType.Input.Dense,
      params = buildParams(),
      activationFunction = Tanh())

  /**
   *
   */
  private fun buildarray1(): DenseNDArray {

    return DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, 0.8, -0.7, -0.5))
  }
  /**
   *
   */
  private fun buildarray2(): DenseNDArray {

    return DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.4, -0.6, -0.2, -0.9))
  }
  /**
   *
   */
  private fun buildarray3(): DenseNDArray {

    return DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, 0.4, 0.2, 0.8))
  }

  /**
   *
   */
  fun buildParams(): NormalizationLayerParameters {

    val params = NormalizationLayerParameters(inputSize = 4)

    params.g.values.assignValues(
        DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, 0.0, -0.3, 0.8))
    )

    params.b.values.assignValues(
        DenseNDArrayFactory.arrayOf(doubleArrayOf(0.9, 0.2, -0.9, 0.2))
    )

    return params
  }

  /**
   *
   */
  fun getOutputErrors1() = DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.0, -0.2, 0.4, 0.0))

  /**
   *
   */
  fun getOutputErrors2() = DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.0, -0.2, 0.4, 0.0))

  /**
   *
   */
  fun getOutputErrors3() = DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.0, -0.2, 0.4, 0.0))

}