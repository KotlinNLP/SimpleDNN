/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.merge.cosinesimilarity

import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.helpers.BackwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The helper which executes the forward on a [CosineLayer].
 *
 * @property layer the layer in which the forward is executed
 */
class CosineBackwardHelper (override val layer: CosineLayer) : BackwardHelper<DenseNDArray> {

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
    val input1Errors = DenseNDArrayFactory.fill(this.layer.inputArray1.values.shape, 0.0)
    val input2Errors = DenseNDArrayFactory.fill(this.layer.inputArray2.values.shape, 0.0)
    val normProd = this.layer.input1Norm * this.layer.input2Norm

    (0 until this.layer.inputArray1.values.length).forEach { i ->

      input1Errors[i] = gy[0] * ((this.layer.inputArray2.values[i] / normProd) -
          ((this.layer.inputArray1.values[i] * this.layer.inputArray1.values[i] * this.layer.inputArray2.values[i]) /
              (normProd * this.layer.input1Norm * this.layer.input1Norm) ) )

      input2Errors[i] = gy[0] * ((this.layer.inputArray1.values[i] / normProd) -
          ((this.layer.inputArray2.values[i] * this.layer.inputArray2.values[i] * this.layer.inputArray1.values[i]) /
              (normProd * this.layer.input2Norm * this.layer.input2Norm) ) )
    }

    this.layer.inputArray1.assignErrors(input1Errors)
    this.layer.inputArray2.assignErrors(input2Errors)

  }
}