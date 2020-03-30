/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.merge.affine

import com.kotlinnlp.simplednn.core.layers.helpers.BackwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the backward on an [AffineLayer].
 *
 * @property layer the layer in which the backward is executed
 */
internal class AffineBackwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: AffineLayer<InputNDArrayType>
) : BackwardHelper<InputNDArrayType>(layer) {

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param propagateToInput whether to propagate the errors to the input array
   */
  override fun execBackward(propagateToInput: Boolean) {

    this.layer.applyOutputActivationDeriv()

    this.assignParamsGradients()

    if (propagateToInput) {
      this.assignLayerGradients()
    }
  }

  /**
   *
   */
  private fun assignParamsGradients() {

    val gy: DenseNDArray = this.layer.outputArray.errors
    val gb: NDArray<*> = this.layer.params.b.errors.values

    gb.assignValues(gy)

    this.layer.inputArrays.zip(this.layer.params.w.map { it.errors }).forEach { (x, gw) ->
      gw.values.assignDot(gy, x.values.t)
    }
  }

  /**
   * Assign the the layer gradients.
   */
  private fun assignLayerGradients() {

    val gy: DenseNDArray = this.layer.outputArray.errors
    val gyT: DenseNDArray = gy.t

    this.layer.inputArrays.zip(this.layer.params.w).forEach { (x, w) ->
      x.assignErrorsByDotT(gyT, w.values)
    }
  }
}
