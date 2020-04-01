/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.attention.scaleddot

import com.kotlinnlp.simplednn.core.arrays.getInputErrors
import com.kotlinnlp.simplednn.core.functionalities.activations.SoftmaxBase
import com.kotlinnlp.simplednn.core.layers.helpers.BackwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The helper which executes the backward on the [ScaledDotAttentionLayer].
 *
 * @property layer the [ScaledDotAttentionLayer] in which the backward is executed
 */
internal class ScaledDotAttentionBackwardHelper(
  override val layer: ScaledDotAttentionLayer
) : BackwardHelper<DenseNDArray>(layer) {

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param propagateToInput whether to propagate the errors to the input array
   */
  override fun execBackward(propagateToInput: Boolean) {

    val outputErrors: Sequence<DenseNDArray> = this.layer.outputArrays.asSequence().map { it.errors }

    this.assignValuesGradients(outputErrors)
    this.assignAttentionGradient(outputErrors)

    this.assignParamsGradients()

    if (propagateToInput) {
      this.assignInputGradients()
    }
  }

  /**
   * Assign the errors of the values.
   *
   * @param outputErrors the errors of the output arrays
   */
  private fun assignValuesGradients(outputErrors: Sequence<DenseNDArray>) {

    val outputErrorsMatrix: DenseNDArray = DenseNDArrayFactory.fromRows(outputErrors.toList())
    val attentionActMatrixT: DenseNDArray = DenseNDArrayFactory.fromColumns(this.layer.attentionAct)

    this.layer.values.assignErrorsByDot(outputErrorsMatrix, attentionActMatrixT)
  }

  /**
   * Assign the errors of the attention matrix and propagate them to the queries and values.
   *
   * @param outputErrors the errors of the output arrays
   */
  private fun assignAttentionGradient(outputErrors: Sequence<DenseNDArray>) {

    val k: DenseNDArray = this.layer.keys.values
    val q: DenseNDArray = this.layer.queries.values
    val vT: DenseNDArray = this.layer.values.values.t

    val attentionErrors: Sequence<DenseNDArray> = outputErrors.map { it.dot(vT) }

    val attentionInnerErrors: DenseNDArray = DenseNDArrayFactory.fromRows(
      this.layer.attentionAct.asSequence().zip(attentionErrors)
        .map { (attention, errors) -> SoftmaxBase().dfOptimized(attention).dot(errors) }
        .toList()
    )

    attentionInnerErrors.assignProd(this.layer.params.attentionFactor)

    this.layer.queries.assignErrorsByDot(attentionInnerErrors, k)
    this.layer.keys.assignErrorsByDot(attentionInnerErrors, q)
  }

  /**
   * Assign the errors of the parameters.
   */
  private fun assignParamsGradients() {

    this.layer.queries.assignParamsGradients(
      gw = this.layer.params.queries.errors.values,
      gb = null,
      x = this.layer.inputMatrix.values)

    this.layer.keys.assignParamsGradients(
      gw = this.layer.params.keys.errors.values,
      gb = null,
      x = this.layer.inputMatrix.values)

    this.layer.values.assignParamsGradients(
      gw = this.layer.params.values.errors.values,
      gb = null,
      x = this.layer.inputMatrix.values)
  }

  /**
   * Assign the errors of the input arrays.
   */
  private fun assignInputGradients() {

    val inputErrors: DenseNDArray =
      this.layer.queries.getInputErrors(w = this.layer.params.queries.values)
        .assignSum(this.layer.keys.getInputErrors(w = this.layer.params.keys.values))
        .assignSum(this.layer.values.getInputErrors(w = this.layer.params.values.values))

    this.layer.inputArrays.asSequence().zip(inputErrors.getRows().asSequence()).forEach { (array, errors) ->
      array.assignErrors(errors)
    }
  }
}
