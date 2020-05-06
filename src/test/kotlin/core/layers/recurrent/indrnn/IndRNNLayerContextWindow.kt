/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.recurrent.indrnn

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.recurrent.RecurrentLayerUnit
import com.kotlinnlp.simplednn.core.layers.models.recurrent.LayerContextWindow
import com.kotlinnlp.simplednn.core.layers.models.recurrent.indrnn.IndRNNLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.recurrent.indrnn.IndRNNLayer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
sealed class IndRNNLayerContextWindow: LayerContextWindow {

  /**
   *
   */
  object Empty : IndRNNLayerContextWindow() {

    override fun getPrevState(): IndRNNLayer<DenseNDArray>? = null

    override fun getNextState(): IndRNNLayer<DenseNDArray>? = null
  }

  /**
   *
   */
  object Back : IndRNNLayerContextWindow() {

    override fun getPrevState(): IndRNNLayer<DenseNDArray> = buildPrevStateLayer()

    override fun getNextState(): IndRNNLayer<DenseNDArray>? = null
  }

  /**
   *
   */
  object Front : IndRNNLayerContextWindow() {

    override fun getPrevState(): IndRNNLayer<DenseNDArray>? = null

    override fun getNextState(): IndRNNLayer<DenseNDArray> = buildNextStateLayer()
  }

  /**
   *
   */
  object Bilateral : IndRNNLayerContextWindow() {

    override fun getPrevState(): IndRNNLayer<DenseNDArray> = buildPrevStateLayer()

    override fun getNextState(): IndRNNLayer<DenseNDArray> = buildNextStateLayer()
  }
}

/**
 *
 */
private fun buildPrevStateLayer(): IndRNNLayer<DenseNDArray> {

  val outputArray = RecurrentLayerUnit<DenseNDArray>(5)
  outputArray.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.2, 0.2, -0.3, -0.9, -0.8)))
  outputArray.setActivation(Tanh)
  outputArray.activate()

  return IndRNNLayer(
    inputArray = AugmentedArray(size = 4),
    inputType = LayerType.Input.Dense,
    outputArray = outputArray,
    params = IndRNNLayerParameters(inputSize = 4, outputSize = 5),
    activationFunction = Tanh,
    layerContextWindow = IndRNNLayerContextWindow.Empty)
}

/**
 *
 */
private fun buildNextStateLayer(): IndRNNLayer<DenseNDArray> {

  val outputArray = RecurrentLayerUnit<DenseNDArray>(5)
  outputArray.assignErrors(errors =  DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.1, -0.5, 0.7, 0.2)))

  return IndRNNLayer(
    inputArray = AugmentedArray(size = 4),
    inputType = LayerType.Input.Dense,
    outputArray = outputArray,
    params = IndRNNLayerParameters(inputSize = 4, outputSize = 5),
    activationFunction = Tanh,
    layerContextWindow = IndRNNLayerContextWindow.Empty)
}
