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
 * @property inputArrays the sequence of attention arrays
 * @property params the parameters of the Attention
 * @param activation the activation function (default SoftmaxBase)
 * @property id an identification number useful to track a specific [AttentionMechanism]
 */
class AttentionMechanism(
  val inputArrays: List<DenseNDArray>,
  val params: AttentionParameters,
  activation: ActivationFunction = SoftmaxBase(),
  override val id: Int = 0
) : ItemsPool.IDItem {

  /**
   * A matrix containing the attention arrays as rows.
   */
  val attentionMatrix: AugmentedArray<DenseNDArray> = AugmentedArray(
    DenseNDArrayFactory.arrayOf(this.inputArrays.map { it.toDoubleArray() })
  )

  /**
   * The array containing the importance score.
   */
  val importanceScore = AugmentedArray<DenseNDArray>(this.inputArrays.size)

  /**
   * The array containing the attention context
   */
  private val attentionContext = AugmentedArray<DenseNDArray>(this.inputArrays.size)

  /**
   * Check requirements.
   */
  init {

    require(this.inputArrays.isNotEmpty()) { "The attention sequence cannot be empty." }
    require(this.inputArrays.all { it.length == this.params.attentionSize }) {
      "The attention arrays must have the expected size (%d).".format(this.params.attentionSize)
    }

    this.importanceScore.setActivation(activation)
  }

  /**
   * Execute the forward of the Attention Mechanism.
   *
   *   am = attention matrix
   *   cv = context vector
   *   ac = attention context
   *
   *   ac = am (dot) cv
   *   importance = activation(ac)
   *
   * @return the importance score
   */
  fun forward(): DenseNDArray {

    this.attentionContext.assignValues(this.attentionMatrix.values.dot(this.params.contextVector.values))
    this.importanceScore.assignValues(this.attentionContext.values)
    this.importanceScore.activate()

    return this.importanceScore.values
  }

  /**
   * Execute the backward assigning the errors of the context vector and the attention matrix.
   *
   *   am = attention matrix
   *   cv = context vector
   *   ac = attention context
   *   gI = importance score errors
   *
   *   gAC = activation'(ac) * gI  // attention context errors
   *   gCV = am (dot) gAC  // context vector errors
   *
   *   gAM = gAC (dot) cv  // attention matrix errors
   *
   * @param paramsErrors the errors of the Attention parameters
   * @param outputErrors the errors of the importance score
   */
  fun backward(paramsErrors: AttentionParameters, outputErrors: DenseNDArray) {

    this.importanceScore.assignErrors(outputErrors)
    this.assignAttentionContextErrors()
    this.assignParamsErrors(paramsErrors)
    this.assignAttentionMatrixErrors()
  }

  /**
   * @return the errors of the attention arrays.
   */
  fun getInputErrors(): List<DenseNDArray> = List(
    size = this.attentionMatrix.values.shape.dim1,
    init = { i -> this.attentionMatrix.errors.getRow(i).t }
  )

  /**
   * Assign the errors of the [attentionContext].
   */
  private fun assignAttentionContextErrors() {

    val gI = this.importanceScore.errors

    this.importanceScore.calculateActivationDeriv().let {

      if (it.isMatrix) // Jacobian matrix
        this.attentionContext.assignErrorsByDot(it, gI)
      else
        this.attentionContext.assignErrorsByProd(it, gI)
    }
  }

  /**
   * @param paramsErrors where to assign the errors of the Attention parameters
   */
  private fun assignParamsErrors(paramsErrors: AttentionParameters) {
    paramsErrors.contextVector.values.assignValues(this.attentionContext.errors.t.dot(this.attentionMatrix.values).t)
  }

  /**
   * Assign the errors of the [attentionMatrix].
   */
  private fun assignAttentionMatrixErrors() {
    this.attentionMatrix.assignErrorsByDot(this.attentionContext.errors, this.params.contextVector.values.t)
  }
}
