/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.structure.utils

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.recurrent.LayerContextWindow
import com.kotlinnlp.simplednn.core.layers.recurrent.deltarnn.DeltaRNNLayerParameters
import com.kotlinnlp.simplednn.core.layers.recurrent.deltarnn.DeltaRNNLayerStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
object DeltaRNNLayerStructureUtils {

  /**
   *
   */
  fun buildLayer(layerContextWindow: LayerContextWindow) = DeltaRNNLayerStructure(
    inputArray = AugmentedArray(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.8, -0.9, -0.9, 1.0))),
    outputArray = AugmentedArray(DenseNDArrayFactory.emptyArray(Shape(5))),
    params = this.buildParams(),
    activationFunction = Tanh(),
    layerContextWindow = layerContextWindow)

  /**
   *
   */
  fun buildParams(): DeltaRNNLayerParameters {

    val params = DeltaRNNLayerParameters(inputSize = 4, outputSize = 5)

    params.feedforwardUnit.weights.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.5, 0.6, -0.8, -0.6),
        doubleArrayOf(0.7, -0.4, 0.1, -0.8),
        doubleArrayOf(0.7, -0.7, 0.3, 0.5),
        doubleArrayOf(0.8, -0.9, 0.0, -0.1),
        doubleArrayOf(0.4, 1.0, -0.7, 0.8)
      )))

    params.feedforwardUnit.biases.values.assignValues(
      DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, 0.0, -0.3, 0.8, -0.4)))

    params.recurrentUnit.weights.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.0, 0.8, 0.8, -1.0, -0.7),
        doubleArrayOf(-0.7, -0.8, 0.2, -0.7, 0.7),
        doubleArrayOf(-0.9, 0.9, 0.7, -0.5, 0.5),
        doubleArrayOf(0.0, -0.1, 0.5, -0.2, -0.8),
        doubleArrayOf(-0.6, 0.6, 0.8, -0.1, -0.3)
      )))

    params.recurrentUnit.biases.values.assignValues(
      DenseNDArrayFactory.arrayOf(doubleArrayOf(0.9, -0.5, 0.4, -0.8, 0.2)))

    params.alpha.values.assignValues(
      DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.5, -0.3, 0.3, 0.4, 0.1)))

    params.beta1.values.assignValues(
      DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.3, -0.4, -0.4, -0.4, -0.4)))

    params.beta2.values.assignValues(
      DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.4, -0.2, 1.0, -0.8, 0.1)))

    return params
  }

  /**
   *
   */
  fun getOutputGold() = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.57, 0.75, -0.15, 1.64, 0.45))
}
