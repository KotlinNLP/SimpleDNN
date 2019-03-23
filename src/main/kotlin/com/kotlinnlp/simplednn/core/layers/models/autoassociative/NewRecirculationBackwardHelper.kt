/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.autoassociative

import com.kotlinnlp.simplednn.core.layers.helpers.BackwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the backward on the [NewRecirculationLayer].
 *
 * @property layer the [NewRecirculationLayer] in which the backward is executed
 */
class NewRecirculationBackwardHelper(
  override val layer: NewRecirculationLayer
) : BackwardHelper<DenseNDArray>(layer) {

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input, starting from
   * the preset errors of the output array.
   *
   *  gb = yI - yR
   *  gw = (yI - yR) (dot) xI + ((xI - xR) (dot) yR)'
   *
   * @param propagateToInput whether to propagate the errors to the input array
   */
  override fun execBackward(propagateToInput: Boolean) {

    this.assignParamsGradients()

    if (propagateToInput) {
      this.assignLayerGradients()
    }
  }

  /**
   * Assign params gradients.
   */
  private fun assignParamsGradients() {

    val xR: DenseNDArray = this.layer.realInput.values
    val yR: DenseNDArray = this.layer.realOutput.values
    val xI: DenseNDArray = this.layer.imaginaryInput.values
    val yI: DenseNDArray = this.layer.imaginaryOutput.values

    val gw: DenseNDArray = this.layer.params.unit.weights.errors.values as DenseNDArray
    val gb: DenseNDArray = this.layer.params.unit.biases.errors.values as DenseNDArray

    val gx: DenseNDArray = xI.sub(xR)
    val gy: DenseNDArray = yI.sub(yR)

    gw.assignDot(gy, xI.t).assignSum(yR.dot(gx.t))
    gb.assignValues(gy)
  }

  /**
   * TODO: assign layer gradients
   */
  private fun assignLayerGradients() { }
}
