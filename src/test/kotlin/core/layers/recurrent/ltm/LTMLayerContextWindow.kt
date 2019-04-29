/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.recurrent.ltm

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.recurrent.LayerContextWindow
import com.kotlinnlp.simplednn.core.layers.models.recurrent.ltm.LTMLayer
import com.kotlinnlp.simplednn.core.layers.models.recurrent.ltm.LTMLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
sealed class LTMLayerContextWindow: LayerContextWindow {

  /**
   *
   */
  class Empty: LTMLayerContextWindow() {

    override fun getPrevState() = null

    override fun getNextState() = null
  }

  /**
   *
   */
  class Back: LTMLayerContextWindow() {

    override fun getPrevState(): LTMLayer<DenseNDArray> = buildPrevStateLayer()

    override fun getNextState() = null
  }

  /**
   *
   */
  class Front(private val refLayer: LTMLayer<DenseNDArray>? = null): LTMLayerContextWindow() {

    override fun getPrevState() = null

    override fun getNextState(): LTMLayer<DenseNDArray> = this.refLayer ?: buildNextStateLayer()
  }

  /**
   *
   */
  class Bilateral: LTMLayerContextWindow() {

    override fun getPrevState(): LTMLayer<DenseNDArray> = buildPrevStateLayer()

    override fun getNextState(): LTMLayer<DenseNDArray> = buildNextStateLayer()
  }
}

/**
 *
 */
private fun buildPrevStateLayer(): LTMLayer<DenseNDArray> {

  val outputArray = AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.2, 0.2, -0.3, -0.9)))
  outputArray.activate()

  val layer = LTMLayer(
    inputArray = AugmentedArray<DenseNDArray>(size = 4),
    inputType = LayerType.Input.Dense,
    outputArray = outputArray,
    params = LTMLayerParameters(inputSize = 4),
    layerContextWindow = LTMLayerContextWindow.Empty())

  layer.cell.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.8, -0.6, 1.0, 0.1)))
  layer.cell.activate()

  return layer
}

/**
 *
 */
private fun buildNextStateLayer(): LTMLayer<DenseNDArray> {

  val outputArray: AugmentedArray<DenseNDArray> = AugmentedArray.zeros(4)
  outputArray.assignErrors(errors = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.1, -0.5, 0.7)))

  val layer = LTMLayer(
    inputArray = AugmentedArray.zeros(size = 4),
    inputType = LayerType.Input.Dense,
    outputArray = outputArray,
    params = LTMLayerParameters(inputSize = 4),
    layerContextWindow = LTMLayerContextWindow.Empty())

  layer.inputArray.assignErrors(errors = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.7, -0.3, -0.2, 0.3)))
  layer.c.assignErrors(errors = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.9, 0.2, -0.5)))

  return layer
}
