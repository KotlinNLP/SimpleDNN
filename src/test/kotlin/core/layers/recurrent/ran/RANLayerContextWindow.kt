/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.recurrent.ran

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.recurrent.LayerContextWindow
import com.kotlinnlp.simplednn.core.layers.models.recurrent.ran.RANLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.recurrent.ran.RANLayer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape

/**
 *
 */
sealed class RANLayerContextWindow: LayerContextWindow {

  /**
   *
   */
  class Empty: RANLayerContextWindow() {

    override fun getPrevState() = null

    override fun getNextState() = null
  }

  /**
   *
   */
  class Back: RANLayerContextWindow() {

    override fun getPrevState(): RANLayer<DenseNDArray> = buildPrevStateLayer()

    override fun getNextState() = null
  }

  /**
   *
   */
  class Front: RANLayerContextWindow() {

    override fun getPrevState() = null

    override fun getNextState(): RANLayer<DenseNDArray> = buildNextStateLayer()
  }

  /**
   *
   */
  class Bilateral: RANLayerContextWindow() {

    override fun getPrevState(): RANLayer<DenseNDArray> = buildPrevStateLayer()

    override fun getNextState(): RANLayer<DenseNDArray> = buildNextStateLayer()
  }
}

/**
 *
 */
private fun buildPrevStateLayer(): RANLayer<DenseNDArray> {

  val outputArray = AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.2, 0.2, -0.3, -0.9, -0.8)))
  outputArray.setActivation(Tanh)
  outputArray.activate()

  return RANLayer(
    inputArray = AugmentedArray(size = 4),
    inputType = LayerType.Input.Dense,
    outputArray = outputArray,
    params = RANLayerParameters(inputSize = 4, outputSize = 5),
    activationFunction = Tanh,
    layerContextWindow = RANLayerContextWindow.Empty()
  )
}

/**
 *
 */
private fun buildNextStateLayer(): RANLayer<DenseNDArray> {

  val outputArray: AugmentedArray<DenseNDArray> = AugmentedArray(values = DenseNDArrayFactory.emptyArray(Shape(5)))
  outputArray.assignErrors(errors = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.1, -0.5, 0.7, 0.2)))

  val layer = RANLayer(
    inputArray = AugmentedArray<DenseNDArray>(size = 4),
    inputType = LayerType.Input.Dense,
    outputArray = outputArray,
    params = RANLayerParameters(inputSize = 4, outputSize = 5),
    activationFunction = Tanh,
    layerContextWindow = RANLayerContextWindow.Empty())

  layer.inputGate.assignValues(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.8, 1.0, -0.8, 0.0, 0.1)))
  layer.inputGate.assignErrors(errors = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.7, -0.3, -0.2, 0.3, 0.6)))
  layer.forgetGate.assignValues(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.2, -0.1, 0.6, -0.8, 0.5)))
  layer.forgetGate.assignErrors(errors = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.9, 0.2, -0.5, 1.0)))

  return layer
}
