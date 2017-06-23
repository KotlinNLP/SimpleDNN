/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.simple

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.BackwardHelper
import com.kotlinnlp.simplednn.core.layers.LayerParameters
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
    val prevStateLayer = this.layer.layerContextWindow.getPrevStateLayer()

    if (nextStateLayer != null) {
      this.addLayerRecurrentGradients(nextStateLayer as SimpleRecurrentLayerStructure<*>)
    }

    this.assignParamsGradients(prevStateLayer?.outputArray)

    if (propagateToInput) {
      this.assignLayerGradients()
    }
  }

  /**
   * gb = gy * 1
   * gw = gb (dot) x
   * gwRec = gy (dot) yPrev
   */
  private fun assignParamsGradients(prevStateOutput: AugmentedArray<DenseNDArray>?) {

    this.layer.outputArray.assignParamsGradients(
      paramsErrors = this.paramsErrors.unit,
      x = this.layer.inputArray.values,
      yPrev = prevStateOutput?.values)
  }

  /**
   * gx = (gb (dot) w) * xDeriv
   */
  private fun assignLayerGradients() { this.layer.params as SimpleRecurrentLayerParameters

    val gx: DenseNDArray = this.layer.inputArray.errors

    // gx = gb (dot) w
    gx.assignValues(this.layer.outputArray.getInputErrors(parameters = this.layer.params.unit))

    // gx *= xDeriv
    if (this.layer.inputArray.hasActivation && gx is DenseNDArray) {
      gx.assignProd(this.layer.inputArray.calculateActivationDeriv())
    }
  }

  /**
   * gy += (gyNext (dot) wRec) * yDeriv
   */
  private fun addLayerRecurrentGradients(nextStateLayer: SimpleRecurrentLayerStructure<*>) {
    this.layer.params as SimpleRecurrentLayerParameters

    val gy: DenseNDArray = this.layer.outputArray.errors

    // gRec = gyNext (dot) wRec
    val gRec: DenseNDArray = nextStateLayer.outputArray.getRecurrentErrors(parameters = this.layer.params.unit)

    // gRec *= yDeriv
    if (this.layer.outputArray.hasActivation) {
      val yDeriv: DenseNDArray = this.layer.outputArray.calculateActivationDeriv()
      gRec.assignProd(yDeriv)
    }

    gy.assignSum(gRec)
  }
}
