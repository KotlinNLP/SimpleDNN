/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.simple

import com.kotlinnlp.simplednn.core.layers.helpers.BackwardHelper
import com.kotlinnlp.simplednn.core.arrays.getInputErrors
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * The helper which executes the backward on a [layer].
 *
 * @property layer the [FeedforwardLayer] in which the backward is executed
 */
class FeedforwardBackwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: FeedforwardLayer<InputNDArrayType>
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
   * gb = gy * 1
   * gw = gy (dot) x
   */
  private fun assignParamsGradients() {

    this.layer.outputArray.assignParamsGradients(
      gw = this.layer.params.unit.weights.errors.values,
      gb = this.layer.params.unit.biases.errors.values,
      x = this.layer.inputArray.values)
  }

  /**
   * gx = gy (dot) w
   */
  private fun assignLayerGradients() {

    this.layer.inputArray.assignErrors(
      errors = this.layer.outputArray.getInputErrors(w = this.layer.params.unit.weights.values)
    )
  }
}
