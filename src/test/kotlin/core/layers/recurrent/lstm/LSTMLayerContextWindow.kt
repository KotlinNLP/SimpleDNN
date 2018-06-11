/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.recurrent.lstm

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.models.recurrent.LayerContextWindow
import com.kotlinnlp.simplednn.core.layers.models.recurrent.lstm.LSTMLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.recurrent.lstm.LSTMLayerStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape

/**
 *
 */
sealed class LSTMLayerContextWindow: LayerContextWindow {

  /**
   *
   */
  class Empty: LSTMLayerContextWindow() {

    override fun getPrevStateLayer() = null

    override fun getNextStateLayer() = null
  }

  /**
   *
   */
  class Back: LSTMLayerContextWindow() {

    override fun getPrevStateLayer(): LSTMLayerStructure<DenseNDArray> = buildPrevStateLayer()

    override fun getNextStateLayer() = null
  }

  /**
   *
   */
  class BackHidden: LSTMLayerContextWindow() {

    private lateinit var initHidden: LSTMLayerStructure<DenseNDArray>

    fun setRefLayer(refLayer: LSTMLayerStructure<DenseNDArray>) {
      this.initHidden = buildInitHiddenLayer(refLayer)
    }

    override fun getPrevStateLayer(): LSTMLayerStructure<DenseNDArray> = this.initHidden

    override fun getNextStateLayer() = null
  }

  /**
   *
   */
  class Front(private val refLayer: LSTMLayerStructure<DenseNDArray>? = null): LSTMLayerContextWindow() {

    override fun getPrevStateLayer() = null

    override fun getNextStateLayer(): LSTMLayerStructure<DenseNDArray> = this.refLayer ?: buildNextStateLayer()
  }

  /**
   *
   */
  class Bilateral: LSTMLayerContextWindow() {

    override fun getPrevStateLayer(): LSTMLayerStructure<DenseNDArray> = buildPrevStateLayer()

    override fun getNextStateLayer(): LSTMLayerStructure<DenseNDArray> = buildNextStateLayer()
  }
}

/**
 *
 */
private fun buildPrevStateLayer(): LSTMLayerStructure<DenseNDArray> {

  val outputArray = AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.2, 0.2, -0.3, -0.9, -0.8)))
  outputArray.activate()

  val layer = LSTMLayerStructure(
    inputArray = AugmentedArray<DenseNDArray>(size = 4),
    outputArray = outputArray,
    params = LSTMLayerParameters(inputSize = 4, outputSize = 5),
    activationFunction = Tanh(),
    layerContextWindow = LSTMLayerContextWindow.Empty())

  layer.cell.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.8, -0.6, 1.0, 0.1, 0.1)))
  layer.cell.activate()

  return layer
}

/**
 *
 */
private fun buildInitHiddenLayer(refLayer: LSTMLayerStructure<DenseNDArray>): LSTMLayerStructure<DenseNDArray> {

  val outputArray = AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.2, 0.2, -0.3, -0.9, -0.8)))

  return LSTMLayerStructure(
    inputArray = AugmentedArray(size = 4),
    outputArray = outputArray,
    params = refLayer.params,
    activationFunction = Tanh(),
    layerContextWindow = LSTMLayerContextWindow.Front(refLayer))
}

/**
 *
 */
private fun buildNextStateLayer(): LSTMLayerStructure<DenseNDArray> {

  val outputArray: AugmentedArray<DenseNDArray> = AugmentedArray(values = DenseNDArrayFactory.emptyArray(Shape(5)))
  outputArray.assignErrors(errors = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.1, -0.5, 0.7, 0.2)))

  val layer = LSTMLayerStructure(
    inputArray = AugmentedArray<DenseNDArray>(size = 4),
    outputArray = outputArray,
    params = LSTMLayerParameters(inputSize = 4, outputSize = 5),
    activationFunction = Tanh(),
    layerContextWindow = LSTMLayerContextWindow.Empty())

  layer.inputGate.assignErrors(errors = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.7, -0.3, -0.2, 0.3, 0.6)))
  layer.outputGate.assignErrors(errors = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.9, 0.2, -0.5, 1.0)))
  layer.forgetGate.assignValues(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.3, -0.4, 0.9, -0.8, -0.4)))
  layer.forgetGate.assignErrors(errors = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.4, 0.6, -0.1, 0.3, 0.0)))
  layer.candidate.assignErrors(errors = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.4, 0.2, -1.0, 0.7, -0.3)))
  layer.cell.assignErrors(errors = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.3, 0.8, 1.0, -0.4, 0.6)))

  return layer
}
