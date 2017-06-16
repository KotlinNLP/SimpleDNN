/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.simple

import com.kotlinnlp.simplednn.core.layers.ForwardHelper
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.LayerStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the forward on a [layer].
 *
 * @property layer the [SimpleRecurrentLayerStructure] in which the forward is executed
 */
class SimpleRecurrentForwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: SimpleRecurrentLayerStructure<InputNDArrayType>
) : ForwardHelper<InputNDArrayType>(layer) {

  /**
   * Forward the input to the output combining it with the parameters.
   *
   * y = f(w (dot) x + b + wRec (dot) yPrev)
   */
  override fun forward() { this.layer.params as SimpleRecurrentLayerParameters

    val w: DenseNDArray = this.layer.params.weights.values as DenseNDArray
    val b: DenseNDArray = this.layer.params.biases.values

    val x: InputNDArrayType = this.layer.inputArray.values
    val y: DenseNDArray = this.layer.outputArray.values

    // y = w (dot) x + b
    y.assignDot(w, x).assignSum(b)

    // y += wRec (dot) yPrev
    val prevStateLayer = this.layer.layerContextWindow.getPrevStateLayer()
    if (prevStateLayer != null) { // recurrent contribute

      val wRec: DenseNDArray = this.layer.params.recurrentWeights.values
      val yPrev: DenseNDArray = prevStateLayer.outputArray.values

      y.assignSum(wRec.dot(yPrev))
    }

    this.layer.outputArray.activate()
  }

  /**
   * Forward the input to the output combining it with the parameters, saving the contributes of the parameters.
   *
   * y = f(w (dot) x + b + wRec (dot) yPrev)
   *
   * @param paramsContributes the [LayerParameters] in which to save the contributes of the input in respect of the
   *                          output
   */
  override fun forward(paramsContributes: LayerParameters) {
    this.layer.params as SimpleRecurrentLayerParameters
    paramsContributes as SimpleRecurrentLayerParameters

    val prevStateLayer: LayerStructure<*>? = this.layer.layerContextWindow.getPrevStateLayer()
    val b: DenseNDArray = this.layer.params.biases.values
    val bContrib: DenseNDArray = if (prevStateLayer != null) b.div(2.0) else b
    // if there's a recurrent contribute b is divided equally within the sum

    // y = w (dot) x + b ( -> b / 2)
    this.forwardArray(
      contributes = paramsContributes.weights.values,
      x = this.layer.inputArray.values,
      y = this.layer.outputArray.values,
      w = this.layer.params.weights.values as DenseNDArray,
      b = bContrib
    )

    // y += wRec (dot) yPrev + b / 2 (recurrent contribute)
    if (prevStateLayer != null) {
      this.addRecurrentContribute(
        prevStateLayer = prevStateLayer,
        paramsContributes = paramsContributes,
        bContribute = bContrib)
    }

    this.layer.outputArray.activate()
  }

  /**
   * Add the recurrent contribute to the output array, saving the contributes of the input in respect of the output.
   *
   * y += wRec (dot) yPrev
   *
   * @param prevStateLayer the layer in the previous state
   * @param paramsContributes the [SimpleRecurrentLayerParameters] in which to save the contributes of the input in
   *                          respect of the output
   * @param bContribute the contribute of the biases
   */
  private fun addRecurrentContribute(prevStateLayer: LayerStructure<*>,
                                     paramsContributes: SimpleRecurrentLayerParameters,
                                     bContribute: DenseNDArray) {

    this.layer.params as SimpleRecurrentLayerParameters

    val y: DenseNDArray = this.layer.outputArray.values
    val yRec: DenseNDArray = paramsContributes.biases.values // a tricky way to save the recurrent contribute
                                                             // (b.size == y.size == yRec.size)
    this.forwardArray(
      contributes = paramsContributes.recurrentWeights.values,
      x = prevStateLayer.outputArray.values,
      y = yRec,
      w = this.layer.params.recurrentWeights.values,
      b = bContribute
    )

    y.assignSum(yRec)
  }
}
