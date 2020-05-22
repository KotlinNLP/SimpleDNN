/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.recurrent.indrnn

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.helpers.BackwardHelper
import com.kotlinnlp.simplednn.core.arrays.getInputErrors
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the backward on a [layer].
 *
 * @property layer the [IndRNNLayer] in which the backward is executed
 */
internal class IndRNNBackwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: IndRNNLayer<InputNDArrayType>
) : BackwardHelper<InputNDArrayType>(layer) {

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param propagateToInput whether to propagate the errors to the input array
   */
  override fun execBackward(propagateToInput: Boolean) {

    val prevStateLayer = this.layer.layersWindow.getPrevState()
    val nextStateLayer = this.layer.layersWindow.getNextState()

    if (nextStateLayer != null) {
      this.addLayerRecurrentGradients(nextStateLayer as IndRNNLayer<*>)
    }

    this.layer.applyOutputActivationDeriv() // must be applied AFTER having added the recurrent contribution

    this.assignParamsGradients(prevStateOutput = prevStateLayer?.outputArray)

    if (propagateToInput) {
      this.assignLayerGradients()
    }
  }

  /**
   * gb = gy * 1
   * gw = gb (dot) x'
   * gwRec = gy * yPrev
   *
   * @param prevStateOutput the output array in the previous state
   */
  private fun assignParamsGradients(prevStateOutput: AugmentedArray<DenseNDArray>?) {

    this.layer.outputArray.assignParamsGradients(
      gw = this.layer.params.feedforwardUnit.weights.errors.values,
      gb = this.layer.params.feedforwardUnit.biases.errors.values,
      x = this.layer.inputArray.values
    )

    val gwRec = this.layer.params.recurrentWeights.errors.values as DenseNDArray
    val yPrev = prevStateOutput?.values

    if (yPrev != null)
      gwRec.assignProd(this.layer.outputArray.errors, yPrev)
    else
      gwRec.zeros()
  }

  /**
   * gx = gb (dot) w
   */
  private fun assignLayerGradients() {

    this.layer.inputArray.assignErrors(
      this.layer.outputArray.getInputErrors(w = this.layer.params.feedforwardUnit.weights.values)
    )
  }

  /**
   * gy += gyNext * wRec
   */
  private fun addLayerRecurrentGradients(nextStateLayer: IndRNNLayer<*>) {

    val gy: DenseNDArray = this.layer.outputArray.errors
    val wRec: DenseNDArray = this.layer.params.recurrentWeights.values
    val gRec = nextStateLayer.outputArray.errors.prod(wRec)

    gy.assignSum(gRec)
  }
}
