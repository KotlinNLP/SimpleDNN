/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.recurrent.lstm

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.recurrent.LayersWindow
import com.kotlinnlp.simplednn.core.layers.models.recurrent.lstm.LSTMLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.recurrent.lstm.LSTMLayer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape

/**
 *
 */
internal sealed class LSTMLayersWindow: LayersWindow {

  /**
   *
   */
  object Empty : LSTMLayersWindow() {

    override fun getPrevState(): Nothing? = null

    override fun getNextState(): Nothing? = null
  }

  /**
   *
   */
  object Back : LSTMLayersWindow() {

    override fun getPrevState(): LSTMLayer<DenseNDArray> = buildPrevStateLayer()

    override fun getNextState(): Nothing? = null
  }

  /**
   *
   */
  class BackHidden: LSTMLayersWindow() {

    private lateinit var initHidden: LSTMLayer<DenseNDArray>

    fun setRefLayer(refLayer: LSTMLayer<DenseNDArray>) {
      this.initHidden = buildInitHiddenLayer(refLayer)
    }

    override fun getPrevState(): LSTMLayer<DenseNDArray> = this.initHidden

    override fun getNextState(): Nothing? = null
  }

  /**
   *
   */
  class Front(private val refLayer: LSTMLayer<DenseNDArray>? = null): LSTMLayersWindow() {

    override fun getPrevState(): Nothing? = null

    override fun getNextState(): LSTMLayer<DenseNDArray> = this.refLayer ?: buildNextStateLayer()
  }

  /**
   *
   */
  object Bilateral : LSTMLayersWindow() {

    override fun getPrevState(): LSTMLayer<DenseNDArray> = buildPrevStateLayer()

    override fun getNextState(): LSTMLayer<DenseNDArray> = buildNextStateLayer()
  }
}

/**
 *
 */
private fun buildPrevStateLayer(): LSTMLayer<DenseNDArray> {

  val outputArray = AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.2, 0.2, -0.3, -0.9, -0.8)))
  outputArray.activate()

  val layer = LSTMLayer(
    inputArray = AugmentedArray<DenseNDArray>(size = 4),
    inputType = LayerType.Input.Dense,
    outputArray = outputArray,
    params = LSTMLayerParameters(inputSize = 4, outputSize = 5),
    activationFunction = Tanh,
    layersWindow = LSTMLayersWindow.Empty)

  layer.cell.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.8, -0.6, 1.0, 0.1, 0.1)))
  layer.cell.activate()

  return layer
}

/**
 *
 */
private fun buildInitHiddenLayer(refLayer: LSTMLayer<DenseNDArray>): LSTMLayer<DenseNDArray> {

  val outputArray = AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.2, 0.2, -0.3, -0.9, -0.8)))

  return LSTMLayer(
    inputArray = AugmentedArray(size = 4),
    inputType = LayerType.Input.Dense,
    outputArray = outputArray,
    params = refLayer.params,
    activationFunction = Tanh,
    layersWindow = LSTMLayersWindow.Front(refLayer))
}

/**
 *
 */
private fun buildNextStateLayer(): LSTMLayer<DenseNDArray> {

  val outputArray: AugmentedArray<DenseNDArray> = AugmentedArray(values = DenseNDArrayFactory.emptyArray(Shape(5)))
  outputArray.assignErrors(errors = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.1, -0.5, 0.7, 0.2)))

  val layer = LSTMLayer(
    inputArray = AugmentedArray<DenseNDArray>(size = 4),
    inputType = LayerType.Input.Dense,
    outputArray = outputArray,
    params = LSTMLayerParameters(inputSize = 4, outputSize = 5),
    activationFunction = Tanh,
    layersWindow = LSTMLayersWindow.Empty)

  layer.inputGate.assignErrors(errors = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.7, -0.3, -0.2, 0.3, 0.6)))
  layer.outputGate.assignErrors(errors = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.9, 0.2, -0.5, 1.0)))
  layer.forgetGate.assignValues(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.3, -0.4, 0.9, -0.8, -0.4)))
  layer.forgetGate.assignErrors(errors = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.4, 0.6, -0.1, 0.3, 0.0)))
  layer.candidate.assignErrors(errors = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.4, 0.2, -1.0, 0.7, -0.3)))
  layer.cell.assignErrors(errors = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.3, 0.8, 1.0, -0.4, 0.6)))

  return layer
}
