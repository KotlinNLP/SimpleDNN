/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.attention.scaleddot

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
    this.assignAttentionGradients(outputErrors)

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

    this.layer.dropoutMaskFull?.let {
      this.layer.values.assignErrors(attentionActMatrixT.dotLeftMasked(outputErrorsMatrix, mask = it))
    } ?: this.layer.values.assignErrorsByDot(attentionActMatrixT, outputErrorsMatrix)

  }

  /**
   * Assign the errors of the attention matrix and propagate them to the queries and values.
   *
   * @param outputErrors the errors of the output arrays
   */
  private fun assignAttentionGradients(outputErrors: Sequence<DenseNDArray>) {

    val k: DenseNDArray = this.layer.keys.values
    val q: DenseNDArray = this.layer.queries.values
    val v: DenseNDArray = this.layer.values.values

    val attentionErrors: Sequence<DenseNDArray> = outputErrors.map { v.dot(it) }

    val attentionInnerErrors: DenseNDArray = DenseNDArrayFactory.fromRows(
      this.layer.attentionAct.asSequence().zip(attentionErrors)
        .mapIndexed { i, (attention, errors) ->
          this.layer.dropoutMasks?.let {
            SoftmaxBase().dfOptimized(attention).dotRightMasked(errors, mask = it[i])
          } ?: SoftmaxBase().dfOptimized(attention).dot(errors)
        }
        .toList()
    )

    attentionInnerErrors.assignProd(this.layer.params.attentionFactor)

    this.layer.queries.assignErrorsByDot(attentionInnerErrors, k)
    this.layer.keys.assignErrorsByDot(attentionInnerErrors.t, q)
  }

  /**
   * Assign the errors of the parameters.
   */
  private fun assignParamsGradients() {

    val x: DenseNDArray = this.layer.inputMatrix.values
    val gQ: DenseNDArray = this.layer.queries.errors
    val gK: DenseNDArray = this.layer.keys.errors
    val gV: DenseNDArray = this.layer.values.errors

    this.layer.params.queries.errors.values.assignDot(gQ.t, x)
    this.layer.params.keys.errors.values.assignDot(gK.t, x)
    this.layer.params.values.errors.values.assignDot(gV.t, x)
  }

  /**
   * Assign the errors of the input arrays.
   */
  private fun assignInputGradients() {

    val gQ: DenseNDArray = this.layer.queries.errors
    val gK: DenseNDArray = this.layer.keys.errors
    val gV: DenseNDArray = this.layer.values.errors

    val wQ: DenseNDArray = this.layer.params.queries.values
    val wK: DenseNDArray = this.layer.params.keys.values
    val wV: DenseNDArray = this.layer.params.values.values

    val inputErrors: DenseNDArray = gQ.dot(wQ).assignSum(gK.dot(wK)).assignSum(gV.dot(wV))

    this.layer.inputArrays.asSequence().zip(inputErrors.getRows().asSequence()).forEach { (array, errors) ->
      array.assignErrors(errors)
    }
  }
}
