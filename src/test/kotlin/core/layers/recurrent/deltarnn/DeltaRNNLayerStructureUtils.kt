/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.recurrent.deltarnn

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.recurrent.LayersWindow
import com.kotlinnlp.simplednn.core.layers.models.recurrent.deltarnn.DeltaRNNLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.recurrent.deltarnn.DeltaRNNLayer
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
internal object DeltaRNNLayerStructureUtils {

  /**
   *
   */
  fun buildLayer(layersWindow: LayersWindow) = DeltaRNNLayer(
    inputArray = AugmentedArray(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.8, -0.9, -0.9, 1.0))),
    inputType = LayerType.Input.Dense,
    outputArray = AugmentedArray(DenseNDArrayFactory.emptyArray(Shape(5))),
    params = buildParams(),
    activationFunction = Tanh,
    layersWindow = layersWindow,
    dropout = 0.0
  )

  /**
   *
   */
  fun buildParams(): DeltaRNNLayerParameters = DeltaRNNLayerParameters(inputSize = 4, outputSize = 5).apply {

    feedforwardUnit.weights.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.5, 0.6, -0.8, -0.6),
        doubleArrayOf(0.7, -0.4, 0.1, -0.8),
        doubleArrayOf(0.7, -0.7, 0.3, 0.5),
        doubleArrayOf(0.8, -0.9, 0.0, -0.1),
        doubleArrayOf(0.4, 1.0, -0.7, 0.8)
      )))

    feedforwardUnit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, 0.0, -0.3, 0.8, -0.4)))

    recurrentUnit.weights.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.0, 0.8, 0.8, -1.0, -0.7),
        doubleArrayOf(-0.7, -0.8, 0.2, -0.7, 0.7),
        doubleArrayOf(-0.9, 0.9, 0.7, -0.5, 0.5),
        doubleArrayOf(0.0, -0.1, 0.5, -0.2, -0.8),
        doubleArrayOf(-0.6, 0.6, 0.8, -0.1, -0.3)
      )))

    recurrentUnit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.9, -0.5, 0.4, -0.8, 0.2)))
    alpha.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.5, -0.3, 0.3, 0.4, 0.1)))
    beta1.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.3, -0.4, -0.4, -0.4, -0.4)))
    beta2.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.4, -0.2, 1.0, -0.8, 0.1)))
  }

  /**
   *
   */
  fun getOutputGold() = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.57, 0.75, -0.15, 1.64, 0.45))
}
