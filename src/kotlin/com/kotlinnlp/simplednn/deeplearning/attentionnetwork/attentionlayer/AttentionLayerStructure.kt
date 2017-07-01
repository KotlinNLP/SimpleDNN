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
 * @property inputSize the size of each array of input
 * @property inputSequence the sequence of input arrays
 * @property attentionSequence the sequence of attention arrays
 */
class AttentionLayerStructure<InputNDArrayType: NDArray<InputNDArrayType>>(
  val inputSequence: ArrayList<AugmentedArray<InputNDArrayType>>,
  val attentionSequence: ArrayList<DenseNDArray>
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
   * The array in which to save the errors of the context vector.
   */
  lateinit var contextVectorErrors: DenseNDArray

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
    require(this.inputSequence.size == this.attentionSequence.size) {
      "The input sequence must have the same length of the attention sequence."
    }

    this.inputSize = this.inputSequence.first().size
    this.checkInputArraysSize() // call it after the inputSize is been set
    this.outputArray = AugmentedArray<DenseNDArray>(values = DenseNDArrayFactory.zeros(Shape(this.inputSize)))
    this.attentionMatrix = this.buildAttentionMatrix()
  }

  /**
   * @return the errors of the attention arrays.
   */
  fun getAttentionErrors(): Array<DenseNDArray> = Array(
    size = this.attentionMatrix.values.shape.dim1,
    init = { i -> this.attentionMatrix.errors.getRow(i).T }
  )

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
