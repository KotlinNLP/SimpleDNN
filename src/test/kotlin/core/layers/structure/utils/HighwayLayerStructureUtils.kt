/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.structure.utils

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.feedforward.highway.HighwayLayerParameters
import com.kotlinnlp.simplednn.core.layers.feedforward.highway.HighwayLayerStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape


/**
 *
 */
object HighwayLayerStructureUtils {

  /**
   *
   */
  fun buildLayer(): HighwayLayerStructure<DenseNDArray> = HighwayLayerStructure(
    inputArray = AugmentedArray(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.8, -0.9, -0.9, 1.0))),
    outputArray = AugmentedArray(DenseNDArrayFactory.emptyArray(Shape(4))),
    params = this.buildParams(),
    activationFunction = Tanh())

  /**
   *
   */
  fun buildParams(): HighwayLayerParameters {

    val params = HighwayLayerParameters(inputSize = 4, outputSize = 4)

    params.input.weights.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.5, 0.6, -0.8, -0.6),
        doubleArrayOf(0.7, -0.4, 0.1, -0.8),
        doubleArrayOf(0.7, -0.7, 0.3, 0.5),
        doubleArrayOf(0.8, -0.9, 0.0, -0.1)
      )))

    params.transformGate.weights.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.1, 0.4, -1.0, 0.4),
        doubleArrayOf(0.7, -0.2, 0.1, 0.0),
        doubleArrayOf(0.7, 0.8, -0.5, -0.3),
        doubleArrayOf(-0.9, 0.9, -0.3, -0.3)
      )))

    params.input.biases.values.assignValues(
      DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, 0.0, -0.3, 0.8))
    )

    params.transformGate.biases.values.assignValues(
      DenseNDArrayFactory.arrayOf(doubleArrayOf(0.9, 0.2, -0.9, 0.2))
    )

    return params
  }

  /**
   *
   */
  fun getOutputErrors() = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.57, 0.75, -0.15, 1.64))
}
