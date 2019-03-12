/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.attention

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.activations.SoftmaxBase
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.utils.ItemsPool

/**
 * The Attention Mechanism.
 *
 * @property attentionSequence the sequence of attention arrays
 * @property params the parameters of the Attention
 * @param activation the activation function (default SoftmaxBase)
 * @property id an identification number useful to track a specific [AttentionMechanism]
 */
open class AttentionMechanism(
  val attentionSequence: List<DenseNDArray>,
  val params: AttentionParameters,
  activation: ActivationFunction = SoftmaxBase(),
  override val id: Int = 0
) : ItemsPool.IDItem {

  /**
   * A matrix containing the attention arrays as rows.
   */
  val attentionMatrix: AugmentedArray<DenseNDArray> = AugmentedArray(
    DenseNDArrayFactory.arrayOf(this.attentionSequence.map { it.toDoubleArray() })
  )

  /**
   * The array containing the importance score.
   */
  val importanceScore = AugmentedArray<DenseNDArray>(this.attentionSequence.size)

  /**
   * Check requirements.
   */
  init {

    require(this.attentionSequence.isNotEmpty()) { "The attention sequence cannot be empty." }
    require(this.attentionSequence.all { it.length == this.params.attentionSize }) {
      "The attention arrays must have the expected size (%d).".format(this.params.attentionSize)
    }

    this.importanceScore.setActivation(activation)
  }

  /**
   * Execute the forward of the Attention Mechanism.
   *
   *   am = attention matrix
   *   cv = context vector
   *
   *   importance = activation(am (dot) cv)
   *
   * @return the importance score
   */
  fun forwardImportanceScore(): DenseNDArray {

    this.importanceScore.assignValues(this.attentionMatrix.values.dot(this.params.contextVector.values))
    this.importanceScore.activate()

    return this.importanceScore.values
  }

  /**
   * Execute the backward assigning the errors of the context vector and the attention matrix.
   *
   *   am = attention matrix
   *   cv = context vector
   *   ac = attention context
   *   gy = output errors
   *   gImportance = importance score errors
   *
   *   gAC = activation'(ac) * gImportance  // attention context errors
   *   gCV = am (dot) gAC  // context vector errors
   *
   *   gAM = gAC (dot) cv  // attention matrix errors
   *
   * @param paramsErrors the errors of the Attention parameters
   * @param importanceScoreErrors the errors of the importance score
   */
  fun backwardImportanceScore(paramsErrors: AttentionParameters, importanceScoreErrors: DenseNDArray) {

    this.assignImportanceScoreErrors(importanceScoreErrors)
    this.assignParamsErrors(paramsErrors)
    this.assignAttentionMatrixErrors()
  }

  /**
   * @return the errors of the attention arrays.
   */
  fun getAttentionErrors(): List<DenseNDArray> = List(
    size = this.attentionMatrix.values.shape.dim1,
    init = { i -> this.attentionMatrix.errors.getRow(i).t }
  )

  /**
   * @param outputErrors the output errors of the importance scores
   */
  private fun assignImportanceScoreErrors(outputErrors: DenseNDArray) {

    val gI = this.importanceScore.calculateActivationDeriv()

    if (gI.isMatrix) // Jacobian matrix
      this.importanceScore.assignErrorsByDot(gI, outputErrors)
    else
      this.importanceScore.assignErrorsByProd(gI, outputErrors)
  }

  /**
   * @param paramsErrors where to assign the errors of the Attention parameters
   */
  private fun assignParamsErrors(paramsErrors: AttentionParameters) {
    paramsErrors.contextVector.values.assignValues(this.importanceScore.errors.t.dot(this.attentionMatrix.values).t)
  }

  /**
   * Assign the errors of the [attentionMatrix].
   */
  private fun assignAttentionMatrixErrors() {
    this.attentionMatrix.assignErrorsByDot(this.importanceScore.errors, this.params.contextVector.values.t)
  }
}
