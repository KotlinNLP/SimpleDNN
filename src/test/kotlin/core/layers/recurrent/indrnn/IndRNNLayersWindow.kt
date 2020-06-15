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
import com.kotlinnlp.simplednn.core.layers.models.recurrent.LayersWindow
import com.kotlinnlp.simplednn.core.layers.models.recurrent.indrnn.IndRNNLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.recurrent.indrnn.IndRNNLayer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
internal sealed class IndRNNLayersWindow: LayersWindow {

  /**
   *
   */
  object Empty : IndRNNLayersWindow() {

    override fun getPrevState(): Nothing? = null

    override fun getNextState(): Nothing? = null
  }

  /**
   *
   */
  object Back : IndRNNLayersWindow() {

    override fun getPrevState(): IndRNNLayer<DenseNDArray> = buildPrevStateLayer()

    override fun getNextState(): Nothing? = null
  }

  /**
   *
   */
  object Front : IndRNNLayersWindow() {

    override fun getPrevState(): Nothing? = null

    override fun getNextState(): IndRNNLayer<DenseNDArray> = buildNextStateLayer()
  }

  /**
   *
   */
  object Bilateral : IndRNNLayersWindow() {

    override fun getPrevState(): IndRNNLayer<DenseNDArray> = buildPrevStateLayer()

    override fun getNextState(): IndRNNLayer<DenseNDArray> = buildNextStateLayer()
  }
}

/**
 *
 */
private fun buildPrevStateLayer(): IndRNNLayer<DenseNDArray> = IndRNNLayer(
  inputArray = AugmentedArray(size = 4),
  inputType = LayerType.Input.Dense,
  outputArray = RecurrentLayerUnit<DenseNDArray>(5).apply {
    assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.2, 0.2, -0.3, -0.9, -0.8)))
    setActivation(Tanh)
    activate()
  },
  params = IndRNNLayerParameters(inputSize = 4, outputSize = 5),
  activationFunction = Tanh,
  layersWindow = IndRNNLayersWindow.Empty,
  dropout = 0.0
)

/**
 *
 */
private fun buildNextStateLayer(): IndRNNLayer<DenseNDArray> = IndRNNLayer(
  inputArray = AugmentedArray(size = 4),
  inputType = LayerType.Input.Dense,
  outputArray = RecurrentLayerUnit<DenseNDArray>(5).apply {
    assignErrors(errors =  DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.1, -0.5, 0.7, 0.2)))
  },
  params = IndRNNLayerParameters(inputSize = 4, outputSize = 5),
  activationFunction = Tanh,
  layersWindow = IndRNNLayersWindow.Empty,
  dropout = 0.0
)
