/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package layers.structure.utils

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.recurrent.LayerContextWindow
import com.kotlinnlp.simplednn.core.layers.recurrent.simple.SimpleRecurrentLayerParameters
import com.kotlinnlp.simplednn.core.layers.recurrent.simple.SimpleRecurrentLayerStructure
import com.kotlinnlp.simplednn.simplemath.NDArray

/**
 *
 */
object SimpleRecurrentLayerStructureUtils {

  /**
   *
   */
  fun buildLayer(layerContextWindow: LayerContextWindow): SimpleRecurrentLayerStructure = SimpleRecurrentLayerStructure(
    inputArray = AugmentedArray(NDArray.arrayOf(doubleArrayOf(-0.8, -0.9, -0.9, 1.0))),
    outputArray = AugmentedArray(size = 5),
    params = this.buildParams(),
    activationFunction = Tanh(),
    layerContextWindow = layerContextWindow)

  /**
   *
   */
  fun buildParams(): SimpleRecurrentLayerParameters {

    val params = SimpleRecurrentLayerParameters(inputSize = 4, outputSize = 5)

    params.weights.values.assignValues(
      NDArray.arrayOf(arrayOf(
        doubleArrayOf(0.5, 0.6, -0.8, -0.6),
        doubleArrayOf(0.7, -0.4, 0.1, -0.8),
        doubleArrayOf(0.7, -0.7, 0.3, 0.5),
        doubleArrayOf(0.8, -0.9, 0.0, -0.1),
        doubleArrayOf(0.4, 1.0, -0.7, 0.8)
      )))

    params.biases.values.assignValues(
      NDArray.arrayOf(doubleArrayOf(0.4, 0.0, -0.3, 0.8, -0.4))
    )

    params.recurrentWeights.values.assignValues(
      NDArray.arrayOf(arrayOf(
        doubleArrayOf(0.0, 0.8, 0.8, -1.0, -0.7),
        doubleArrayOf(-0.7, -0.8, 0.2, -0.7, 0.7),
        doubleArrayOf(-0.9, 0.9, 0.7, -0.5, 0.5),
        doubleArrayOf(0.0, -0.1, 0.5, -0.2, -0.8),
        doubleArrayOf(-0.6, 0.6, 0.8, -0.1, -0.3)
      )))

    return params
  }

  /**
   *
   */
  fun getOutputGold() = NDArray.arrayOf(doubleArrayOf(0.57, 0.75, -0.15, 1.64, 0.45))

}
