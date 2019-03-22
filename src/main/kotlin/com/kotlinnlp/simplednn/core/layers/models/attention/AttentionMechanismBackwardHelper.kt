/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.attention

import com.kotlinnlp.simplednn.core.layers.helpers.BackwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the backward on the [AttentionMechanismLayer].
 *
 * @property layer the [AttentionMechanismLayer] in which the backward is executed
 */
class AttentionMechanismBackwardHelper(
  override val layer: AttentionMechanismLayer
) : BackwardHelper<DenseNDArray>(layer) {

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param propagateToInput whether to propagate the errors to the input array
   */
  override fun execBackward(propagateToInput: Boolean) {

    this.layer.applyOutputActivationDeriv()

    this.assignParamsGradients()
    this.assignAttentionMatrixErrors()

    if (propagateToInput) {
      this.assignLayerGradients()
    }
  }

  /**
   *
   */
  private fun assignParamsGradients() { this.layer.params as AttentionMechanismLayerParameters

    val gCV = this.layer.params.contextVector.errors.values

    gCV.assignValues(this.layer.outputArray.errors.t.dot(this.layer.attentionMatrix.values).t)
  }

  /**
   * Assign the errors of the attentionMatrix.
   */
  private fun assignAttentionMatrixErrors() { this.layer.params as AttentionMechanismLayerParameters

    this.layer.attentionMatrix.assignErrorsByDot(
      this.layer.outputArray.errors, this.layer.params.contextVector.values.t)
  }

  /**
   *
   */
  private fun assignLayerGradients() {

    this.layer.inputArrays.forEachIndexed { i, inputArray ->
      inputArray.assignErrors(this.layer.attentionMatrix.errors.getRow(i).t)
    }
  }
}
