/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.recurrent.deltarnn

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.types.recurrent.RecurrentLayerUnit
import com.kotlinnlp.simplednn.core.layers.types.recurrent.LayerContextWindow
import com.kotlinnlp.simplednn.core.layers.types.recurrent.deltarnn.DeltaRNNLayerParameters
import com.kotlinnlp.simplednn.core.layers.types.recurrent.deltarnn.DeltaRNNLayerStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import core.layers.recurrent.simple.SimpleRecurrentLayerContextWindow

/**
 *
 */
sealed class DeltaLayerContextWindow: LayerContextWindow {

  /**
   *
   */
  class Empty: DeltaLayerContextWindow() {

    override fun getPrevStateLayer() = null

    override fun getNextStateLayer() = null
  }

  /**
   *
   */
  class Back: DeltaLayerContextWindow() {

    override fun getPrevStateLayer(): DeltaRNNLayerStructure<DenseNDArray> = buildPrevStateLayer()

    override fun getNextStateLayer() = null
  }

  /**
   *
   */
  class Front: DeltaLayerContextWindow() {

    override fun getPrevStateLayer() = null

    override fun getNextStateLayer(): DeltaRNNLayerStructure<DenseNDArray> = buildNextStateLayer()
  }

  /**
   *
   */
  class Bilateral: DeltaLayerContextWindow() {

    override fun getPrevStateLayer(): DeltaRNNLayerStructure<DenseNDArray> = buildPrevStateLayer()

    override fun getNextStateLayer(): DeltaRNNLayerStructure<DenseNDArray> = buildNextStateLayer()
  }
}

/**
 *
 */
private fun buildPrevStateLayer(): DeltaRNNLayerStructure<DenseNDArray> {

  val outputArray = RecurrentLayerUnit<DenseNDArray>(5)
  outputArray.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.2, 0.2, -0.3, -0.9, -0.8)))
  outputArray.setActivation(Tanh())
  outputArray.activate()

  return DeltaRNNLayerStructure(
    inputArray = AugmentedArray(size = 4),
    outputArray = outputArray,
    params = DeltaRNNLayerParameters(inputSize = 4, outputSize = 5),
    activationFunction = Tanh(),
    layerContextWindow = DeltaLayerContextWindow.Empty())
}

/**
 *
 */
private fun buildNextStateLayer(): DeltaRNNLayerStructure<DenseNDArray> {

  val outputArray = RecurrentLayerUnit<DenseNDArray>(5)
  outputArray.assignErrors(errors =  DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.1, -0.5, 0.7, 0.2)))

  val layer = DeltaRNNLayerStructure(
    inputArray = AugmentedArray<DenseNDArray>(size = 4),
    outputArray = outputArray,
    params = DeltaRNNLayerParameters(inputSize = 4, outputSize = 5),
    activationFunction = Tanh(),
    layerContextWindow = SimpleRecurrentLayerContextWindow.Empty())

  layer.wx.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.7, -0.7, -0.2, 0.8, -0.6)))
  layer.partition.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.2, -0.1, 0.6, -0.8, 0.5)))
  layer.candidate.assignErrors(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.4, 0.6, -0.1, 0.3, 0.0)))

  return layer
}
