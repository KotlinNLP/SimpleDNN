/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.recurrent.ltm

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.recurrent.LayerContextWindow
import com.kotlinnlp.simplednn.core.layers.models.recurrent.ltm.LTMLayer
import com.kotlinnlp.simplednn.core.layers.models.recurrent.ltm.LTMLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
object LTMLayerStructureUtils {

  /**
   *
   */
  fun buildLayer(layerContextWindow: LayerContextWindow): LTMLayer<DenseNDArray> = LTMLayer(
    inputArray = AugmentedArray(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.8, -0.9, -0.9, 1.0))),
    inputType = LayerType.Input.Dense,
    outputArray = AugmentedArray.zeros(4),
    params = buildParams(),
    layerContextWindow = layerContextWindow)

  /**
   *
   */
  fun buildParams(): LTMLayerParameters {

    val params = LTMLayerParameters(inputSize = 4)

    params.inputGate1.weights.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.5, 0.6, -0.8, -0.6),
        doubleArrayOf(0.7, -0.4, 0.1, -0.8),
        doubleArrayOf(0.7, -0.7, 0.3, 0.5),
        doubleArrayOf(0.8, -0.9, 0.0, -0.1)
      )))

    params.inputGate2.weights.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.1, 0.4, -1.0, 0.4),
        doubleArrayOf(0.7, -0.2, 0.1, 0.0),
        doubleArrayOf(0.7, 0.8, -0.5, -0.3),
        doubleArrayOf(-0.9, 0.9, -0.3, -0.3)
      )))

    params.inputGate3.weights.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(-1.0, 0.2, 0.0, 0.2),
        doubleArrayOf(-0.7, 0.7, -0.3, -0.3),
        doubleArrayOf(0.3, -0.6, 0.0, 0.7),
        doubleArrayOf(-1.0, -0.6, 0.9, 0.8)
      )))

    params.cell.weights.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.2, 0.6, 0.0, 0.1),
      doubleArrayOf(0.1, -0.3, -0.8, -0.5),
      doubleArrayOf(-0.1, 0.0, 0.4, -0.4),
      doubleArrayOf(-0.8, -0.3, -0.7, 0.3)
      )))

    params.inputGate1.biases.values.assignValues(
      DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, 0.0, -0.3, 0.8))
    )

    params.inputGate2.biases.values.assignValues(
      DenseNDArrayFactory.arrayOf(doubleArrayOf(0.9, 0.2, -0.9, 0.2))
    )

    params.inputGate3.biases.values.assignValues(
      DenseNDArrayFactory.arrayOf(doubleArrayOf(0.5, -0.5, 1.0, 0.4)))

    params.cell.biases.values.assignValues(
      DenseNDArrayFactory.arrayOf(doubleArrayOf(0.2, -0.9, -0.9, 0.5)))

    return params
  }

  /**
   *
   */
  fun getOutputGold() = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.57, 0.75, -0.15, 1.64))
}
