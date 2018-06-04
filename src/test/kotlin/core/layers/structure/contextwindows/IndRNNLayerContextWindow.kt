/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.structure.contextwindows

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.RecurrentLayerUnit
import com.kotlinnlp.simplednn.core.layers.recurrent.LayerContextWindow
import com.kotlinnlp.simplednn.core.layers.recurrent.indrnn.IndRNNLayerParameters
import com.kotlinnlp.simplednn.core.layers.recurrent.indrnn.IndRNNLayerStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
sealed class IndRNNLayerContextWindow: LayerContextWindow {

  /**
   *
   */
  class Empty: IndRNNLayerContextWindow() {

    override fun getPrevStateLayer(): IndRNNLayerStructure<DenseNDArray>? = null

    override fun getNextStateLayer(): IndRNNLayerStructure<DenseNDArray>? = null
  }

  /**
   *
   */
  class Back: IndRNNLayerContextWindow() {

    override fun getPrevStateLayer(): IndRNNLayerStructure<DenseNDArray> = buildPrevStateLayer()

    override fun getNextStateLayer(): IndRNNLayerStructure<DenseNDArray>? = null
  }

  /**
   *
   */
  class Front: IndRNNLayerContextWindow() {

    override fun getPrevStateLayer(): IndRNNLayerStructure<DenseNDArray>? = null

    override fun getNextStateLayer(): IndRNNLayerStructure<DenseNDArray> = buildNextStateLayer()
  }

  /**
   *
   */
  class Bilateral: IndRNNLayerContextWindow() {

    override fun getPrevStateLayer(): IndRNNLayerStructure<DenseNDArray> = buildPrevStateLayer()

    override fun getNextStateLayer(): IndRNNLayerStructure<DenseNDArray> = buildNextStateLayer()
  }
}

/**
 *
 */
private fun buildPrevStateLayer(): IndRNNLayerStructure<DenseNDArray> {

  val outputArray = RecurrentLayerUnit<DenseNDArray>(5)
  outputArray.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.2, 0.2, -0.3, -0.9, -0.8)))
  outputArray.setActivation(Tanh())
  outputArray.activate()

  return IndRNNLayerStructure(
    inputArray = AugmentedArray(size = 4),
    outputArray = outputArray,
    params = IndRNNLayerParameters(inputSize = 4, outputSize = 5),
    activationFunction = Tanh(),
    layerContextWindow = IndRNNLayerContextWindow.Empty())
}

/**
 *
 */
private fun buildNextStateLayer(): IndRNNLayerStructure<DenseNDArray> {

  val outputArray = RecurrentLayerUnit<DenseNDArray>(5)
  outputArray.assignErrors(errors =  DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.1, -0.5, 0.7, 0.2)))

  return IndRNNLayerStructure(
    inputArray = AugmentedArray(size = 4),
    outputArray = outputArray,
    params = IndRNNLayerParameters(inputSize = 4, outputSize = 5),
    activationFunction = Tanh(),
    layerContextWindow = IndRNNLayerContextWindow.Empty())
}
