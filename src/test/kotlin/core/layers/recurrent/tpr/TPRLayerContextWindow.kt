/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.recurrent.tpr

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.recurrent.LayerContextWindow
import com.kotlinnlp.simplednn.core.layers.models.recurrent.lstm.LSTMLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.recurrent.tpr.TPRLayer
import com.kotlinnlp.simplednn.core.layers.models.recurrent.tpr.TPRLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
sealed class TPRLayerContextWindow: LayerContextWindow {

  /**
   *
   */
  class Empty: TPRLayerContextWindow() {

    override fun getPrevState() = null

    override fun getNextState() = null
  }

  /**
   *
   */
  class Back: TPRLayerContextWindow() {

    override fun getPrevState(): TPRLayer<DenseNDArray> = buildPrevStateLayer()

    override fun getNextState() = null
  }

  /**
   *
   */
  class Front(private val refLayer: TPRLayer<DenseNDArray>? = null): TPRLayerContextWindow() {

    override fun getPrevState() = null

    override fun getNextState(): TPRLayer<DenseNDArray> = this.refLayer ?: buildNextStateLayer()
  }

  /**
   *
   */
  class Bilateral: TPRLayerContextWindow() {

    override fun getPrevState(): TPRLayer<DenseNDArray> = buildPrevStateLayer()

    override fun getNextState(): TPRLayer<DenseNDArray> = buildNextStateLayer()
  }
}

/**
 *
 */
private fun buildPrevStateLayer(): TPRLayer<DenseNDArray> {

  val layer = TPRLayer(
      inputArray = AugmentedArray<DenseNDArray>(size = 4),
      inputType = LayerType.Input.Dense,
      params = TPRLayerParameters(inputSize = 4, dRoles = 3, dSymbols = 2, nRoles = 3, nSymbols = 4),
      layerContextWindow = TPRLayerContextWindow.Empty(),
      quantizationRegularizer = 0.001)

  layer.outputArray.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.211, -0.451, 0.499, -1.333, -0.11645, 0.366)))

  return layer
}

/**
 *
 */
private fun buildNextStateLayer(): TPRLayer<DenseNDArray> {

  val layer = TPRLayer(
      inputArray = AugmentedArray<DenseNDArray>(size = 4),
      inputType = LayerType.Input.Dense,
      params = TPRLayerParameters(inputSize = 4, dRoles = 3, dSymbols = 2, nRoles = 3, nSymbols = 4),
      layerContextWindow = TPRLayerContextWindow.Empty(),
      quantizationRegularizer = 0.001)

  layer.outputArray.assignErrors(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.711, -0.099, 0.459, -1.235, -0.9845, 0.9292)))

  layer.aS.assignValues(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.8, 1.0, -0.8, 0.0)))
  layer.aS.assignErrors(errors = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.7, 0.3, 0.2, 0.3)))
  layer.aR.assignValues(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.2, -0.1, 0.6)))
  layer.aR.assignErrors(errors = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.9, 0.2)))


  return layer
}
