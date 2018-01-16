/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attentionnetwork.attentionlayer

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
 */
class AttentionLayerStructure<InputNDArrayType: NDArray<InputNDArrayType>>(
  val inputSequence: ArrayList<AugmentedArray<InputNDArrayType>>,
  val attentionSequence: ArrayList<DenseNDArray>,
  val params: AttentionLayerParameters
) {

  /**
   * The output dense array.
   */
  val outputArray: AugmentedArray<DenseNDArray>

  /**
   * A matrix containing the attention arrays as rows.
   */
  val attentionMatrix: AugmentedArray<DenseNDArray>

  /**
   * The array containing the importance score for each element of the input sequence.
   */
  lateinit var importanceScore: DenseNDArray

  /**
   * The size of each array of input.
   */
  private val inputSize: Int

  /**
   * Initialize values.
   */
  init {
    require(this.inputSequence.size > 0) { "The input sequence cannot be empty." }
    require(this.attentionSequence.size > 0) { "The attention sequence cannot be empty." }
    require(this.attentionSequence.all { it.length == this.params.attentionSize }) {
      "The attention arrays must have the expected size (%d).".format(this.params.attentionSize)
    }
    require(this.inputSequence.size == this.attentionSequence.size) {
      "The input sequence must have the same length of the attention sequence."
    }

    this.inputSize = this.inputSequence.first().size
    this.checkInputArraysSize() // call it after the inputSize is been set
    this.outputArray = AugmentedArray(values = DenseNDArrayFactory.zeros(Shape(this.inputSize)))
    this.attentionMatrix = this.buildAttentionMatrix()
  }

  /**
   * @return the errors of the attention arrays.
   */
  fun getAttentionErrors(): Array<DenseNDArray> = Array(
    size = this.attentionMatrix.values.shape.dim1,
    init = { i -> this.attentionMatrix.errors.getRow(i).t }
  )

  /**
   * Set the errors of the [outputArray].
   *
   * @param errors the errors to set into the outputArray
   */
  fun setErrors(errors: DenseNDArray) = this.outputArray.assignErrors(errors)

  /**
   * Perform the forward of the input sequence.
   *
   * @return the output array
   */
  fun forward(): DenseNDArray {

    val helper = AttentionLayerForwardHelper(layer = this)

    helper.forward()

    return this.outputArray.values
  }

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the errors of the output array.
   *
   * @param paramsErrors the errors of the parameters which will be filled
   * @param propagateToInput whether to propagate the errors to the input sequence
   */
  fun backward(paramsErrors: AttentionLayerParameters, propagateToInput: Boolean) {

    val helper = AttentionLayerBackwardHelper(layer = this)

    helper.backward(paramsErrors = paramsErrors, propagateToInput = propagateToInput)
  }

  /**
   * Check the size of the input arrays.
   */
  private fun checkInputArraysSize() {
    this.inputSequence.forEach {
      require(it.size == this.inputSize)
    }
  }

  /**
   * Build the attention matrix.
   */
  private fun buildAttentionMatrix() = AugmentedArray(
    values = DenseNDArrayFactory.arrayOf(
      Array(
        size = this.attentionSequence.size,
        init = { i -> this.attentionSequence[i].toDoubleArray() }
      )
    )
  )
}
