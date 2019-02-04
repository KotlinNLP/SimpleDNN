/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.attention

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.SoftmaxBase
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.utils.ItemsPool

/**
 * The Attention Mechanism.
 *
 * @property attentionSequence the sequence of attention arrays
 * @property params the parameters of the Attention
 * @property id an identification number useful to track a specific [AttentionMechanism]
 */
open class AttentionMechanism(
  val attentionSequence: List<DenseNDArray>,
  val params: AttentionParameters,
  override val id: Int = 0
) : ItemsPool.IDItem {

  /**
   * A matrix containing the attention arrays as rows.
   */
  val attentionMatrix: AugmentedArray<DenseNDArray> = AugmentedArray(
    values = DenseNDArrayFactory.arrayOf(this.attentionSequence.map { it.toDoubleArray() })
  )

  /**
   * The array containing the importance score.
   */
  lateinit var importanceScore: DenseNDArray

  /**
   * Check requirements.
   */
  init {

    require(this.attentionSequence.isNotEmpty()) { "The attention sequence cannot be empty." }
    require(this.attentionSequence.all { it.length == this.params.attentionSize }) {
      "The attention arrays must have the expected size (%d).".format(this.params.attentionSize)
    }
  }

  /**
   * Perform the forward of the Attention Mechanism.
   *
   *   am = attention matrix
   *   cv = context vector
   *
   *   ac = am (dot) cv  // attention context
   *   alpha = softmax(ac)  // importance score
   *
   * @return the importance score
   */
  fun forwardImportanceScore(): DenseNDArray {

    val contextVector: DenseNDArray = this.params.contextVector.values
    val attentionContext: DenseNDArray = this.attentionMatrix.values.dot(contextVector)

    this.importanceScore = SoftmaxBase().f(attentionContext)

    return this.importanceScore
  }

  /**
   * Executes the backward assigning the errors of the context vector and the attention matrix.
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
   * @param paramsErrors the errors of the Attention parameters
   * @param importanceScoreErrors the errors of the importance score
   */
  fun backwardImportanceScore(paramsErrors: AttentionParameters, importanceScoreErrors: DenseNDArray) {

    val contextVector: DenseNDArray = this.params.contextVector.values
    val softmaxGradients: DenseNDArray = SoftmaxBase().dfOptimized(this.importanceScore)
    val acErrors: DenseNDArray = softmaxGradients.dot(importanceScoreErrors)

    paramsErrors.contextVector.values.assignValues(acErrors.t.dot(this.attentionMatrix.values).t)

    this.attentionMatrix.assignErrorsByDot(acErrors, contextVector.t)
  }

  /**
   * @return the errors of the attention arrays.
   */
  fun getAttentionErrors(): List<DenseNDArray> = List(
    size = this.attentionMatrix.values.shape.dim1,
    init = { i -> this.attentionMatrix.errors.getRow(i).t }
  )
}
