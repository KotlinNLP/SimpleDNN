/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.simple

import com.kotlinnlp.simplednn.core.layers.helpers.BackwardHelper
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArrayMask

/**
 * The helper which executes the backward on a [layer].
 *
 * @property layer the [FeedforwardLayerStructure] in which the backward is executed
 */
class FeedforwardBackwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: FeedforwardLayerStructure<InputNDArrayType>
) : BackwardHelper<InputNDArrayType> {

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param paramsErrors the errors of the parameters which will be filled
   * @param propagateToInput whether to propagate the errors to the input array
   * @param mePropK the k factor of the 'meProp' algorithm to propagate from the k (in percentage) output nodes with
   *                the top errors (ignored if null, the default)
   */
  override fun backward(paramsErrors: LayerParameters<*>, propagateToInput: Boolean, mePropK: Double?) {

    this.layer.applyOutputActivationDeriv()

    // Be careful: the mask must be get after applying the output activation derivative.
    val outputMask: NDArrayMask? = if (mePropK != null) this.layer.outputArray.getMePropMask(mePropK) else null

    this.assignParamsGradients(paramsErrors as FeedforwardLayerParameters, mePropMask = outputMask)

    if (propagateToInput) {
      this.assignLayerGradients(mePropMask = outputMask)
    }
  }

  /**
   * gb = gy * 1
   * gw = gy (dot) x
   *
   * @param paramsErrors the errors of the parameters which will be filled
   * @param mePropMask the mask of the k output nodes with the top errors
   */
  private fun assignParamsGradients(paramsErrors: FeedforwardLayerParameters, mePropMask: NDArrayMask?) {

    this.layer.outputArray.assignParamsGradients(
      paramsErrors = paramsErrors.unit,
      x = this.layer.inputArray.values,
      mePropMask = mePropMask)
  }

  /**
   * gx = gy (dot) w
   *
   * @param mePropMask the mask of the k output nodes with the top errors
   */
  private fun assignLayerGradients(mePropMask: NDArrayMask?) {

    this.layer.params as FeedforwardLayerParameters

    this.layer.inputArray.assignErrors(
      errors = this.layer.outputArray.getInputErrors(parameters = this.layer.params.unit, mePropMask = mePropMask)
    )
  }
}
