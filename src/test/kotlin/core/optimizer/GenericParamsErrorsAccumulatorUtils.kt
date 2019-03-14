/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.optimizer

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
object GenericParamsErrorsAccumulatorUtils {

  /**
   *
   */
  fun buildEmptyParams(): List<ParamsArray> {

    val params = FeedforwardLayerParameters(inputSize = 4, outputSize = 2)

    params.unit.weights.values.assignValues(DenseNDArrayFactory.arrayOf(listOf(
      doubleArrayOf(0.0, 0.0, 0.0, 0.0),
      doubleArrayOf(0.0, 0.0, 0.0, 0.0)
    )))

    params.unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(
      doubleArrayOf(0.0, 0.0)
    ))

    return params.paramsList.map { ParamsArray(it as UpdatableDenseArray) }
  }

  /**
   *
   */
  fun buildWeightsErrorsValues1() = DenseNDArrayFactory.arrayOf(listOf(
    doubleArrayOf(0.3, 0.4, 0.2, -0.2),
    doubleArrayOf(0.2, -0.1, 0.1, 0.6)
  ))

  /**
   *
   */
  fun buildBiasesErrorsValues1() = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.3, -0.4))

  /**
   *
   */
  fun buildWeightsErrorsValues2() = DenseNDArrayFactory.arrayOf(listOf(
    doubleArrayOf(0.7, -0.8, 0.1, -0.6),
    doubleArrayOf(0.8, 0.6, -0.9, -0.2)
  ))

  /**
   *
   */
  fun buildBiasesErrorsValues2() = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.9, 0.1))
}