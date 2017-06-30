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
  val inputSize: Int,
  val inputSequence: ArrayList<AugmentedArray<InputNDArrayType>>,
  val attentionSequence: ArrayList<DenseNDArray>
) {

  /**
   * The output dense array.
   */
  val outputArray = AugmentedArray<DenseNDArray>(values = DenseNDArrayFactory.zeros(Shape(this.inputSize)))

  /**
   * A matrix containing the attention arrays as rows.
   */
  lateinit var attentionMatrix: AugmentedArray<DenseNDArray>

  /**
   * An array which contains the result of the dot product between the [attentionMatrix] and the context vector.
   */
  lateinit var attentionContext: DenseNDArray

  /**
   * The array containing the importance score for each element of the input sequence.
   */
  lateinit var importanceScore: DenseNDArray

  /**
   * The array in which to save the errors of the context vector.
   */
  lateinit var contextVectorErrors: DenseNDArray
    private set

  /**
   * Initialize values.
   */
  init {
    this.initAttentionMatrix(this.attentionSequence)
  }

  /**
   * Assign the context vector errors.
   *
   * @param errors the errors to assign
   */
  fun assignContextVectorErrors(errors: DenseNDArray) {

    try {
      this.contextVectorErrors.assignValues(errors)

    } catch (e: UninitializedPropertyAccessException) {
      this.contextVectorErrors = errors.copy()
    }
  }

  /**
   *
   */
  fun getAttentionErrors(): Array<DenseNDArray> = Array(
    size = this.attentionMatrix.values.shape.dim1,
    init = { i -> this.attentionMatrix.errors.getRow(i).T }
  )

  /**
   * Init attention matrix values.
   *
   * @param attentionSequence the list of attention arrays of the sequence.
   */
  private fun initAttentionMatrix(attentionSequence: ArrayList<DenseNDArray>) {

    this.attentionMatrix = AugmentedArray(values = DenseNDArrayFactory.arrayOf(
      Array(
        size = attentionSequence.size,
        init = { i ->
          attentionSequence[i].toDoubleArray()
        }
      )
    ))
  }
}
