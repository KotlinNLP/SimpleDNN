/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.recurrent.cfn

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.recurrent.LayersWindow
import com.kotlinnlp.simplednn.core.layers.models.recurrent.cfn.CFNLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.recurrent.cfn.CFNLayer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape

/**
 *
 */
internal sealed class CFNLayersWindow: LayersWindow {

  /**
   *
   */
  object Empty : CFNLayersWindow() {

    override fun getPrevState(): Nothing? = null

    override fun getNextState(): Nothing? = null
  }

  /**
   *
   */
  object Back : CFNLayersWindow() {

    override fun getPrevState(): CFNLayer<DenseNDArray> = buildPrevStateLayer()

    override fun getNextState(): Nothing? = null
  }

  /**
   *
   */
  class Front(val currentLayerOutput: DenseNDArray): CFNLayersWindow() {

    override fun getPrevState(): Nothing? = null

    override fun getNextState(): CFNLayer<DenseNDArray> = buildNextStateLayer(currentLayerOutput)
  }

  /**
   *
   */
  class Bilateral(val currentLayerOutput: DenseNDArray): CFNLayersWindow() {

    override fun getPrevState(): CFNLayer<DenseNDArray> = buildPrevStateLayer()

    override fun getNextState(): CFNLayer<DenseNDArray> = buildNextStateLayer(currentLayerOutput)
  }
}

/**
 *
 */
private fun buildPrevStateLayer(): CFNLayer<DenseNDArray> = CFNLayer(
  inputArray = AugmentedArray(size = 4),
  inputType = LayerType.Input.Dense,
  outputArray = AugmentedArray(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.2, 0.2, -0.3, -0.9, -0.8))).apply {
    activate()
  },
  params = CFNLayerParameters(inputSize = 4, outputSize = 5),
  activationFunction = Tanh,
  layersWindow = CFNLayersWindow.Empty,
  dropout = 0.0
)

/**
 *
 */
private fun buildNextStateLayer(currentLayerOutput: DenseNDArray): CFNLayer<DenseNDArray> = CFNLayer(
  inputArray = AugmentedArray<DenseNDArray>(size = 4),
  inputType = LayerType.Input.Dense,
  outputArray = AugmentedArray(DenseNDArrayFactory.emptyArray(Shape(5))).apply {
    assignErrors(errors = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.1, -0.5, 0.7, 0.2)))
  },
  params = CFNLayerParameters(inputSize = 4, outputSize = 5),
  activationFunction = Tanh,
  layersWindow = CFNLayersWindow.Empty,
  dropout = 0.0
).apply {
  inputGate.assignValues(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.8, 1.0, -0.8, 0.0, 0.1)))
  inputGate.assignErrors(errors = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.7, -0.3, -0.2, 0.3, 0.6)))
  forgetGate.assignValues(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.2, -0.1, 0.6, -0.8, 0.5)))
  forgetGate.assignErrors(errors = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.9, 0.2, -0.5, 1.0)))
  activatedPrevOutput = currentLayerOutput
}
