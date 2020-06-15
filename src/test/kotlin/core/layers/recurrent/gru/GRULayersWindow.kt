/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.recurrent.gru

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.recurrent.LayersWindow
import com.kotlinnlp.simplednn.core.layers.models.recurrent.gru.GRULayerParameters
import com.kotlinnlp.simplednn.core.layers.models.recurrent.gru.GRULayer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape

/**
 *
 */
internal sealed class GRULayersWindow: LayersWindow {

  /**
   *
   */
  object Empty : GRULayersWindow() {

    override fun getPrevState(): Nothing? = null

    override fun getNextState(): Nothing? = null
  }

  /**
   *
   */
  object Back : GRULayersWindow() {

    override fun getPrevState(): GRULayer<DenseNDArray> = buildPrevStateLayer()

    override fun getNextState(): Nothing? = null
  }

  /**
   *
   */
  object Front : GRULayersWindow() {

    override fun getPrevState(): Nothing? = null

    override fun getNextState(): GRULayer<DenseNDArray> = buildNextStateLayer()
  }

  /**
   *
   */
  object Bilateral : GRULayersWindow() {

    override fun getPrevState(): GRULayer<DenseNDArray> = buildPrevStateLayer()

    override fun getNextState(): GRULayer<DenseNDArray> = buildNextStateLayer()
  }
}

/**
 *
 */
private fun buildPrevStateLayer(): GRULayer<DenseNDArray> = GRULayer(
  inputArray = AugmentedArray(size = 4),
  inputType = LayerType.Input.Dense,
  outputArray = AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.2, 0.2, -0.3, -0.9, -0.8))).apply {
    activate()
  },
  params = GRULayerParameters(inputSize = 4, outputSize = 5),
  activationFunction = Tanh,
  layersWindow = GRULayersWindow.Empty,
  dropout = 0.0
)

/**
 *
 */
private fun buildNextStateLayer(): GRULayer<DenseNDArray> = GRULayer(
  inputArray = AugmentedArray<DenseNDArray>(size = 4),
  inputType = LayerType.Input.Dense,
  outputArray = AugmentedArray(values = DenseNDArrayFactory.emptyArray(Shape(5))).apply {
    assignErrors(errors = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.1, -0.5, 0.7, 0.2)))
  },
  params = GRULayerParameters(inputSize = 4, outputSize = 5),
  activationFunction = Tanh,
  layersWindow = GRULayersWindow.Empty,
  dropout = 0.0
).apply {
  resetGate.assignValues(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.8, 1.0, -0.8, 0.0, 0.1)))
  resetGate.assignErrors(errors = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.7, -0.3, -0.2, 0.3, 0.6)))
  partitionGate.assignValues(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.2, -0.1, 0.6, -0.8, 0.5)))
  partitionGate.assignErrors(errors = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.9, 0.2, -0.5, 1.0)))
  candidate.assignErrors(errors = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.4, 0.6, -0.1, 0.3, 0.0)))
}
