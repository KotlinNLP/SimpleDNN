/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package layers.structure.contextwindows

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.recurrent.LayerContextWindow
import com.kotlinnlp.simplednn.core.layers.recurrent.simple.SimpleRecurrentLayerParameters
import com.kotlinnlp.simplednn.core.layers.recurrent.simple.SimpleRecurrentLayerStructure
import com.kotlinnlp.simplednn.simplemath.NDArray

/**
 *
 */
sealed class SimpleRecurrentLayerContextWindow: LayerContextWindow {

  /**
   *
   */
  class Empty: SimpleRecurrentLayerContextWindow() {

    override fun getPrevStateLayer() = null

    override fun getNextStateLayer() = null
  }

  /**
   *
   */
  class Back: SimpleRecurrentLayerContextWindow() {

    override fun getPrevStateLayer(): SimpleRecurrentLayerStructure = buildPrevStateLayer()

    override fun getNextStateLayer() = null
  }

  /**
   *
   */
  class Front: SimpleRecurrentLayerContextWindow() {

    override fun getPrevStateLayer() = null

    override fun getNextStateLayer(): SimpleRecurrentLayerStructure = buildNextStateLayer()
  }
}

/**
 *
 */
private fun buildPrevStateLayer(): SimpleRecurrentLayerStructure {

  val outputArray = com.kotlinnlp.simplednn.core.arrays.AugmentedArray(NDArray.arrayOf(doubleArrayOf(-0.2, 0.2, -0.3, -0.9, -0.8)))
  outputArray.activate()

  return SimpleRecurrentLayerStructure(
    inputArray = AugmentedArray(size = 4),
    outputArray = outputArray,
    params = SimpleRecurrentLayerParameters(inputSize = 4, outputSize = 5),
    activationFunction = Tanh(),
    layerContextWindow = SimpleRecurrentLayerContextWindow.Empty())
}

/**
 *
 */
private fun buildNextStateLayer(): SimpleRecurrentLayerStructure {

  val outputArray = com.kotlinnlp.simplednn.core.arrays.AugmentedArray(size = 5)
  outputArray.assignErrors(NDArray.arrayOf(doubleArrayOf(0.1, 0.1, -0.5, 0.7, 0.2)))

  val layer = SimpleRecurrentLayerStructure(
    inputArray = AugmentedArray(size = 4),
    outputArray = outputArray,
    params = SimpleRecurrentLayerParameters(inputSize = 4, outputSize = 5),
    activationFunction = Tanh(),
    layerContextWindow = SimpleRecurrentLayerContextWindow.Empty())

  return layer
}
