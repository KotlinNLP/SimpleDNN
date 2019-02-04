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
 * @property activation the activation function (default SoftmaxBase)
 * @property id an identification number useful to track a specific [AttentionMechanism]
 */
open class AttentionMechanism(
  val attentionSequence: List<DenseNDArray>,
  val params: AttentionParameters,
  private val activation: ActivationFunction = SoftmaxBase(),
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
   * Execute the forward of the Attention Mechanism.
   *
   *   am = attention matrix
   *   cv = context vector
   *
   *   ac = am (dot) cv  // attention context
   *   importance = activation(ac)
   *
   * @return the importance score
   */
  fun forwardImportanceScore(): DenseNDArray {

    val contextVector: DenseNDArray = this.params.contextVector.values
    val attentionContext: DenseNDArray = this.attentionMatrix.values.dot(contextVector)

    this.importanceScore = this.activation.f(attentionContext)

    return this.importanceScore
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

    val contextVector: DenseNDArray = this.params.contextVector.values
    val activationGradients: DenseNDArray = this.activation.dfOptimized(this.importanceScore)

    val acErrors: DenseNDArray = if (activationGradients.isMatrix) // e.g. softmax jacobian
      activationGradients.dot(importanceScoreErrors)
    else
      activationGradients.prod(importanceScoreErrors)

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
