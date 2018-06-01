/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.attention

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The structure of the Attention Layer.
 *
 * @property inputSequence the sequence of input arrays
 * @property attentionSequence the sequence of attention arrays
 * @property params the parameters of the Attention Layer
 * @property id an identification number useful to track a specific [AttentionLayerStructure]
 */
class AttentionLayerStructure<InputNDArrayType: NDArray<InputNDArrayType>>(
  val inputSequence: ArrayList<AugmentedArray<InputNDArrayType>>,
  attentionSequence: ArrayList<DenseNDArray>,
  params: AttentionParameters,
  id: Int = 0
) : AttentionMechanismStructure(attentionSequence = attentionSequence, params = params, id = id) {

  /**
   * The output dense array.
   */
  val outputArray: AugmentedArray<DenseNDArray>

  /**
   * The size of each array of input.
   */
  private val inputSize: Int

  /**
   * Initialize values.
   */
  init {
    require(this.inputSequence.size > 0) { "The input sequence cannot be empty." }
    require(this.inputSequence.size == attentionSequence.size) {
      "The input sequence must have the same length of the attention sequence."
    }

    this.inputSize = this.inputSequence.first().size
    this.checkInputArraysSize() // call it after the inputSize is been set
    this.outputArray = AugmentedArray(values = DenseNDArrayFactory.zeros(Shape(this.inputSize)))
  }

  /**
   * Set the errors of the [outputArray].
   *
   * @param errors the errors to set into the outputArray
   */
  fun setOutputErrors(errors: DenseNDArray) = this.outputArray.assignErrors(errors)

  /**
   * Perform the forward of the input sequence.
   *
   * @return the output array
   */
  fun forward(): DenseNDArray {

    super.forwardImportanceScore()

    this.calculateOutput()

    return this.outputArray.values
  }

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the errors of the output array.
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
   * @param paramsErrors the errors of the parameters which will be filled
   * @param propagateToInput whether to propagate the errors to the input sequence
   */
  fun backward(paramsErrors: AttentionParameters, propagateToInput: Boolean) {

    super.backwardImportanceScore(paramsErrors = paramsErrors, importanceScoreErrors = this.getScoreErrors())

    if (propagateToInput) {
      this.setInputErrors()
    }
  }

  /**
   * Calculate the values of the output array.
   *
   *   y = sum by { x_i * alpha_i }
   */
  private fun calculateOutput() {

    val y: DenseNDArray = this.outputArray.values

    y.zeros()

    this.inputSequence.forEachIndexed { i, inputArray ->
      y.assignSum(inputArray.values.prod(this.importanceScore[i]))
    }
  }

  /**
   * gScore_i = x_i' (dot) gy
   *
   * @return the errors of the importance score array.
   */
  private fun getScoreErrors(): DenseNDArray {

    val outputErrors: DenseNDArray = this.outputArray.errors
    val scoreErrors: DenseNDArray = DenseNDArrayFactory.zeros(shape = Shape(this.inputSequence.size))

    for (i in 0 until this.inputSequence.size) {
      val inputArray = this.inputSequence[i].values
      scoreErrors[i] = inputArray.prod(outputErrors).sum()
    }

    return scoreErrors
  }

  /**
   * Set the errors of each array of the input sequence (which is into the structure).
   *
   *   gx_i = gy * alpha_i  // errors of the i-th input array
   */
  private fun setInputErrors() {

    val outputErrors: DenseNDArray = this.outputArray.errors
    val score: DenseNDArray = this.importanceScore

    for (i in 0 until this.inputSequence.size) {
      this.inputSequence[i].assignErrorsByProd(outputErrors, score[i])
    }
  }

  /**
   * Check the size of the input arrays.
   */
  private fun checkInputArraysSize() {
    this.inputSequence.forEach {
      require(it.size == this.inputSize)
    }
  }
}
