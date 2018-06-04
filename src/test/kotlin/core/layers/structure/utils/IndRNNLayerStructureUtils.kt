/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.structure.utils

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.RecurrentLayerUnit
import com.kotlinnlp.simplednn.core.layers.recurrent.LayerContextWindow
import com.kotlinnlp.simplednn.core.layers.recurrent.indrnn.IndRNNLayerParameters
import com.kotlinnlp.simplednn.core.layers.recurrent.indrnn.IndRNNLayerStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
object IndRNNLayerStructureUtils {

  /**
   *
   */
  fun buildLayer(layerContextWindow: LayerContextWindow) = IndRNNLayerStructure(
    inputArray = AugmentedArray(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.8, -0.9, -0.9, 1.0))),
    outputArray = RecurrentLayerUnit(5),
    params = this.buildParams(),
    activationFunction = Tanh(),
    layerContextWindow = layerContextWindow)

  /**
   *
   */
  fun buildParams(): IndRNNLayerParameters {

    val params = IndRNNLayerParameters(inputSize = 4, outputSize = 5)

    params.feedforwardUnit.weights.values.assignValues(
      DenseNDArrayFactory.arrayOf(arrayOf(
        doubleArrayOf(0.5, 0.6, -0.8, -0.6),
        doubleArrayOf(0.7, -0.4, 0.1, -0.8),
        doubleArrayOf(0.7, -0.7, 0.3, 0.5),
        doubleArrayOf(0.8, -0.9, 0.0, -0.1),
        doubleArrayOf(0.4, 1.0, -0.7, 0.8)
      )))

    params.feedforwardUnit.biases.values.assignValues(
      DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, 0.0, -0.3, 0.8, -0.4))
    )

    params.recurrentWeights.values.assignValues(
      DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, -0.7, -0.9, 0.0, -0.6))
    )

    return params
  }

  /**
   *
   */
  fun getOutputErrors() = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.57, 0.75, -0.15, 1.64, 0.45))
}
