/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.recurrent.deltarnn

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.recurrent.RecurrentLayerUnit
import com.kotlinnlp.simplednn.core.layers.models.recurrent.LayersWindow
import com.kotlinnlp.simplednn.core.layers.models.recurrent.deltarnn.DeltaRNNLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.recurrent.deltarnn.DeltaRNNLayer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
internal sealed class DeltaLayersWindow: LayersWindow {

  /**
   *
   */
  object Empty : DeltaLayersWindow() {

    override fun getPrevState(): Nothing? = null

    override fun getNextState(): Nothing? = null
  }

  /**
   *
   */
  object Back : DeltaLayersWindow() {

    override fun getPrevState(): DeltaRNNLayer<DenseNDArray> = buildPrevStateLayer()

    override fun getNextState(): Nothing? = null
  }

  /**
   *
   */
  object Front : DeltaLayersWindow() {

    override fun getPrevState(): Nothing? = null

    override fun getNextState(): DeltaRNNLayer<DenseNDArray> = buildNextStateLayer()
  }

  /**
   *
   */
  object Bilateral : DeltaLayersWindow() {

    override fun getPrevState(): DeltaRNNLayer<DenseNDArray> = buildPrevStateLayer()

    override fun getNextState(): DeltaRNNLayer<DenseNDArray> = buildNextStateLayer()
  }
}

/**
 *
 */
private fun buildPrevStateLayer(): DeltaRNNLayer<DenseNDArray> = DeltaRNNLayer(
  inputArray = AugmentedArray(size = 4),
  inputType = LayerType.Input.Dense,
  outputArray = RecurrentLayerUnit<DenseNDArray>(5).apply {
    assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.2, 0.2, -0.3, -0.9, -0.8)))
    setActivation(Tanh)
    activate()
  },
  params = DeltaRNNLayerParameters(inputSize = 4, outputSize = 5),
  activationFunction = Tanh,
  layersWindow = DeltaLayersWindow.Empty,
  dropout = 0.0
)

/**
 *
 */
private fun buildNextStateLayer(): DeltaRNNLayer<DenseNDArray> = DeltaRNNLayer(
  inputArray = AugmentedArray<DenseNDArray>(size = 4),
  inputType = LayerType.Input.Dense,
  outputArray = RecurrentLayerUnit<DenseNDArray>(5).apply {
    assignErrors(errors = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.1, -0.5, 0.7, 0.2)))
  },
  params = DeltaRNNLayerParameters(inputSize = 4, outputSize = 5),
  activationFunction = Tanh,
  layersWindow = DeltaLayersWindow.Empty,
  dropout = 0.0
).apply {
  wx.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.7, -0.7, -0.2, 0.8, -0.6)))
  partition.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.2, -0.1, 0.6, -0.8, 0.5)))
  candidate.assignErrors(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.4, 0.6, -0.1, 0.3, 0.0)))
}
