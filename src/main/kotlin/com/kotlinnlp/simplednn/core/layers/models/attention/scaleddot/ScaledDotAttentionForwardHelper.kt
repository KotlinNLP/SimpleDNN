/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.attention.scaleddot

import com.kotlinnlp.simplednn.core.functionalities.activations.SoftmaxBase
import com.kotlinnlp.simplednn.core.layers.helpers.ForwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the forward on a [layer].
 *
 * @property layer the [ScaledDotAttentionLayer] in which the forward is executed
 */
internal class ScaledDotAttentionForwardHelper(
  override val layer: ScaledDotAttentionLayer
) : ForwardHelper<DenseNDArray>(layer) {

  /**
   * Forward the input to the output combining it with the parameters.
   *
   *   A = Softmax((Q (dot) K') / sqrt(dk))
   *   Y = A (dot) V
   */
  override fun forward() {

    this.forwardInputs()

    val q: DenseNDArray = this.layer.queries.values
    val k: DenseNDArray = this.layer.keys.values
    val vT: DenseNDArray = this.layer.values.values.t

    this.layer.attention = q.dot(k.t).assignProd(this.layer.params.attentionFactor)
    this.layer.attentionAct = this.layer.attention.getRows().map { SoftmaxBase().f(it).t }

    this.layer.outputArrays.zip(this.layer.attentionAct).forEachIndexed { i, (y, a) ->
      this.layer.dropoutMasks?.let {
        y.assignValues(vT.dotRightMasked(a, mask = it[i]))
      } ?: y.assignValues(vT.dot(a))
    }
  }

  /**
   * Forward the input to calculate queries, keys and values.
   *
   *   Q = Wq (dot) I + Bq
   *   K = Wk (dot) I + Bk
   *   V = Wv (dot) I + Bv
   */
  private fun forwardInputs() {

    /**
     * Add the same bias [b] to each row of [m].
     */
    fun addBias(m: DenseNDArray, b: DenseNDArray) {
      (0 until m.rows).forEach { i ->
        (0 until m.columns).forEach { j ->
          m[i, j] += b[j]
        }
      }
    }

    val x: DenseNDArray = this.layer.inputMatrix.values

    val wQ: DenseNDArray = this.layer.params.queries.weights.values
    val wK: DenseNDArray = this.layer.params.keys.weights.values
    val wV: DenseNDArray = this.layer.params.values.weights.values

    this.layer.queries.assignValues(x.dot(wQ.t))
    this.layer.keys.assignValues(x.dot(wK.t))
    this.layer.values.assignValues(x.dot(wV.t))

    addBias(m = this.layer.queries.values, b = this.layer.params.queries.biases.values)
    addBias(m = this.layer.keys.values, b = this.layer.params.keys.biases.values)
    addBias(m = this.layer.values.values, b = this.layer.params.values.biases.values)
  }
}
