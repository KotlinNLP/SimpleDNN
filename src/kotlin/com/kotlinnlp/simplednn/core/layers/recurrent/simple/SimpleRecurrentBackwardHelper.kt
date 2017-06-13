/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.simple

import com.kotlinnlp.simplednn.core.layers.BackwardHelper
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.LayerStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the backward on a [layer].
 *
 * @property layer the [SimpleRecurrentLayerStructure] in which the backward is executed
 */
class SimpleRecurrentBackwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: SimpleRecurrentLayerStructure<InputNDArrayType>
) : BackwardHelper<InputNDArrayType> {

  /**
   * A support variable to manage the errors on the parameters during the backward
   */
  lateinit private var paramsErrors: SimpleRecurrentLayerParameters

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param paramsErrors the errors of the parameters which will be filled
   * @param propagateToInput whether to propagate the errors to the input array
   */
  override fun backward(paramsErrors: LayerParameters, propagateToInput: Boolean) {

    this.paramsErrors = paramsErrors as SimpleRecurrentLayerParameters

    val nextStateLayer = this.layer.layerContextWindow.getNextStateLayer()

    if (nextStateLayer != null) {
      this.addLayerRecurrentGradients(nextStateLayer)
    }

    this.assignParamsGradients(nextStateLayer)

    if (propagateToInput) {
      this.assignLayerGradients()
    }
  }

  /**
   * gb = gy * 1
   * gw = gb (dot) x
   * gwRec = gyNext (dot) y
   */
  private fun assignParamsGradients(nextStateLayer: LayerStructure<*>?) {

    val x: InputNDArrayType = this.layer.inputArray.values
    val gb: DenseNDArray = this.paramsErrors.biases.values
    val gw: NDArray<*> = this.paramsErrors.weights.values
    val gy: DenseNDArray = this.layer.outputArray.errors

    gb.assignValues(gy)
    gw.assignDot(gb, x.T)

    if (nextStateLayer != null) { // recurrent errors
      val gyNext: DenseNDArray = nextStateLayer.outputArray.errors
      val y: DenseNDArray = this.layer.outputArray.values
      val gwRec: DenseNDArray = this.paramsErrors.recurrentWeights.values

      gwRec.assignDot(gyNext, y.T)
    }
  }

  /**
   * gx = (gb (dot) w) * xDeriv
   */
  private fun assignLayerGradients() { this.layer.params as SimpleRecurrentLayerParameters

    val gb: DenseNDArray = this.paramsErrors.biases.values
    val w: DenseNDArray = this.layer.params.weights.values as DenseNDArray

    val gx: DenseNDArray = this.layer.inputArray.errors

    // gx = gb (dot) w
    gx.assignValues(gb.T.dot(w))

    // gx *= xDeriv
    if (this.layer.inputArray.hasActivation && gx is DenseNDArray) {
      gx.assignProd(this.layer.inputArray.calculateActivationDeriv())
    }
  }

  /**
   * gy += (gyNext (dot) wRec) * yDeriv
   */
  private fun addLayerRecurrentGradients(nextStateLayer: LayerStructure<*>) {
    this.layer.params as SimpleRecurrentLayerParameters

    val gy: DenseNDArray = this.layer.outputArray.errors
    val gyNext: DenseNDArray = nextStateLayer.outputArray.errors
    val wRec: DenseNDArray = this.layer.params.recurrentWeights.values

    // gRec = gyNext (dot) wRec
    val gRec: DenseNDArray = gyNext.T.dot(wRec)

    // gRec *= yDeriv
    if (this.layer.outputArray.hasActivation) {
      val yDeriv: DenseNDArray = this.layer.outputArray.calculateActivationDeriv()
      gRec.assignProd(yDeriv)
    }

    gy.assignSum(gRec)
  }
}
