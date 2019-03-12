/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.merge.distance

import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.helpers.BackwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The helper which executes the forward on a [DistanceLayer].
 *
 * @property layer the layer in which the forward is executed
 */
class DistanceBackwardHelper(override val layer: DistanceLayer) : BackwardHelper<DenseNDArray> {

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param paramsErrors the errors of the parameters which will be filled
   * @param propagateToInput whether to propagate the errors to the input array
   */
  override fun backward(paramsErrors: LayerParameters<*>, propagateToInput: Boolean) {

    if (propagateToInput) {
      this.assignLayerGradients()
    }
  }

  /**
   * Assign the the layer gradients.
   */
  private fun assignLayerGradients() {

    val gy: DenseNDArray = this.layer.outputArray.errors

    gy.assignProd(this.layer.outputArray.values)

    val scoreError = gy[0]

    val input1Errors = DenseNDArrayFactory.fill(this.layer.inputArray1.values.shape, scoreError)
    val input2Errors = DenseNDArrayFactory.fill(this.layer.inputArray2.values.shape, scoreError)

    (0 until this.layer.inputArray1.values.length).forEach { i ->

      when {
        this.layer.inputArray1.values[i] > this.layer.inputArray2.values[i] -> input1Errors[i] *= -1.0
        this.layer.inputArray1.values[i] < this.layer.inputArray2.values[i] -> input2Errors[i] *= -1.0
        else -> {
          input1Errors[i] = 0.0
          input2Errors[i] = 0.0
        }
      }

    }

    this.layer.inputArray1.assignErrors(input1Errors)
    this.layer.inputArray2.assignErrors(input2Errors)
  }
}