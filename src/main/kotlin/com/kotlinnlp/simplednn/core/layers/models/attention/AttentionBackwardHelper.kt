/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.attention

import com.kotlinnlp.simplednn.core.layers.helpers.BackwardHelper
import com.kotlinnlp.simplednn.core.layers.models.attention.attentionmechanism.AttentionMechanismLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the backward on the [AttentionLayer].
 *
 * @property layer the [AttentionLayer] in which the backward is executed
 */
internal class AttentionBackwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: AttentionLayer<InputNDArrayType>
) : BackwardHelper<InputNDArrayType>(layer) {

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   *   x_i = i-th input array
   *   alpha_i = i-th value of alpha
   *   am = attention matrix
   *   gy = output errors
   *
   *   gScore_i = x_i' (dot) gy
   *   gAC = softmax_jacobian(alpha) (dot) gScore  // attention context errors
   *   gCV = am (dot) gAC  // context vector errors
   *
   *   gAM = gAC (dot) cv  // attention matrix errors
   *   gx_i = gy * alpha_i  // errors of the i-th input array
   *
   * @param propagateToInput whether to propagate the errors to the input array
   */
  override fun execBackward(propagateToInput: Boolean) { this.layer.params as AttentionMechanismLayerParameters

    this.setAttentionErrors()

    this.layer.attentionMechanism.backward(propagateToInput = true)
    this.touchParamsErrors() // important

    if (propagateToInput) {
      this.assignLayerGradients()
    }
  }

  /**
   * Set the output errors of the attention mechanism.
   */
  private fun setAttentionErrors() {

    val gy: DenseNDArray = this.layer.outputArray.errors

    this.layer.attentionMechanism.outputArray.assignZeroErrors()

    this.layer.attentionMechanism.outputArray.errors.let { attentionErrors ->

      for (i in 0 until attentionErrors.length) {
        attentionErrors[i] = this.layer.inputArrays[i].values.prod(gy).sum()
      }
    }
  }

  /**
   * Touch the context vector params so that it can be returned from the backward() method.
   */
  private fun touchParamsErrors() { (this.layer.params as AttentionMechanismLayerParameters).contextVector.errors }

  /**
   * Set the errors of each input array.
   *
   *   gx_i = gy * alpha_i  // errors of the i-th input array
   */
  private fun assignLayerGradients() {

    val outputErrors: DenseNDArray = this.layer.outputArray.errors
    val attentionScores: DenseNDArray = this.layer.attentionMechanism.outputArray.values

    this.layer.inputArrays.indices.forEach { i ->
      this.layer.inputArrays[i].assignErrorsByProd(outputErrors, attentionScores[i])
    }
  }
}
