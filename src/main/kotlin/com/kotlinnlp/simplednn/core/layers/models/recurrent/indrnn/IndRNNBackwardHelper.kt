/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.recurrent.indrnn

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.helpers.BackwardHelper
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.assignParamsGradients
import com.kotlinnlp.simplednn.core.layers.getInputErrors
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the backward on a [layer].
 *
 * @property layer the [IndRNNLayer] in which the backward is executed
 */
class IndRNNBackwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: IndRNNLayer<InputNDArrayType>
) : BackwardHelper<InputNDArrayType> {

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param paramsErrors the errors of the parameters which will be filled
   * @param propagateToInput whether to propagate the errors to the input array
   */
  override fun backward(paramsErrors: LayerParameters<*>, propagateToInput: Boolean) {

    val prevStateLayer = this.layer.layerContextWindow.getPrevState()
    val nextStateLayer = this.layer.layerContextWindow.getNextState()

    if (nextStateLayer != null) {
      this.addLayerRecurrentGradients(nextStateLayer as IndRNNLayer<*>)
    }

    this.layer.applyOutputActivationDeriv() // must be applied AFTER having added the recurrent contribution

    this.assignParamsGradients(
      paramsErrors = paramsErrors as IndRNNLayerParameters,
      prevStateOutput = prevStateLayer?.outputArray)

    if (propagateToInput) {
      this.assignLayerGradients()
    }
  }

  /**
   * gb = gy * 1
   * gw = gb (dot) x'
   * gwRec = gy * yPrev
   *
   * @param paramsErrors the errors of the parameters which will be filled
   * @param prevStateOutput the output array in the previous state
   */
  private fun assignParamsGradients(paramsErrors: IndRNNLayerParameters,
                                    prevStateOutput: AugmentedArray<DenseNDArray>?) {

    this.layer.outputArray.assignParamsGradients(
      gw = paramsErrors.feedforwardUnit.weights.values,
      gb = paramsErrors.feedforwardUnit.biases.values,
      x = this.layer.inputArray.values
    )

    val gwRec = paramsErrors.recurrentWeights.values as DenseNDArray
    val yPrev = prevStateOutput?.values

    if (yPrev != null)
      gwRec.assignProd(this.layer.outputArray.errors, yPrev)
    else
      gwRec.zeros()
  }

  /**
   * gx = gb (dot) w
   */
  private fun assignLayerGradients() { this.layer.params as IndRNNLayerParameters

    this.layer.inputArray.assignErrors(
      this.layer.outputArray.getInputErrors(w = this.layer.params.feedforwardUnit.weights.values)
    )
  }

  /**
   * gy += gyNext * wRec
   */
  private fun addLayerRecurrentGradients(nextStateLayer: IndRNNLayer<*>) {

    // TODO: mePropMask

    this.layer.params as IndRNNLayerParameters

    val gy: DenseNDArray = this.layer.outputArray.errors
    val wRec: DenseNDArray = this.layer.params.recurrentWeights.values as DenseNDArray
    val gRec = nextStateLayer.outputArray.errors.prod(wRec)

    gy.assignSum(gRec)
  }
}
