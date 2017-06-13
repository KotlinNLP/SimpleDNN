/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.feedforward

import com.kotlinnlp.simplednn.core.layers.BackwardHelper
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the backward on a [layer].
 *
 * @property layer the [FeedforwardLayerStructure] in which the backward is executed
 */
class FeedforwardBackwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: FeedforwardLayerStructure<InputNDArrayType>
) : BackwardHelper<InputNDArrayType> {

  /**
   * A support variable to manage the errors on the parameters during the backward
   */
  lateinit private var paramsErrors: FeedforwardLayerParameters

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param paramsErrors the errors of the parameters which will be filled
   * @param propagateToInput whether to propagate the errors to the input array
   */
  override fun backward(paramsErrors: LayerParameters, propagateToInput: Boolean) {

    this.paramsErrors = paramsErrors as FeedforwardLayerParameters

    this.assignParamsGradients()

    if (propagateToInput) {
      this.assignLayerGradients()
    }
  }

  /**
   * gb = gy * 1
   * gw = gy (dot) x
   */
  private fun assignParamsGradients() {

    val gb: DenseNDArray = this.paramsErrors.biases.values
    val gw: NDArray<*> = this.paramsErrors.weights.values

    val x: InputNDArrayType = this.layer.inputArray.values
    val gy: DenseNDArray = this.layer.outputArray.errors

    gb.assignValues(gy)
    gw.assignDot(gy, x.T)
  }

  /**
   * gx = (gy (dot) w) * xDeriv
   */
  private fun assignLayerGradients() { this.layer.params as FeedforwardLayerParameters

    val gy: DenseNDArray = this.layer.outputArray.errors
    val w: DenseNDArray = this.layer.params.weights.values as DenseNDArray

    val gx: DenseNDArray = this.layer.inputArray.errors

    gx.assignValues(gy.T.dot(w))

    if (this.layer.inputArray.hasActivation && gx is DenseNDArray) {
      gx.assignProd(this.layer.inputArray.calculateActivationDeriv())
    }
  }
}
