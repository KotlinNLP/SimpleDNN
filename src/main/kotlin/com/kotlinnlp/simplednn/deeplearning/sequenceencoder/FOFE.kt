/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.sequenceencoder

import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.toMatrix
import kotlin.math.pow

/**
 * Fixed-size Ordinally-Forgetting Encoding (Zhang et al., 2015b).
 *
 * @param alpha the forgetting factor (0 < α ≤ 0.5)
 */
class FOFE(
  val alpha: Double,
  override val id: Int = 0
) : NeuralProcessor<
  List<DenseNDArray>, // InputType
  List<DenseNDArray>, // OutputType
  List<DenseNDArray>, // ErrorsType
  List<DenseNDArray>  // InputErrorsType
  > {

  companion object {

    /**
     * Build a T-order lower triangular matrix.
     * Each row vector of the matrix represents a FOFE code of the partial sequence.
     */
    private fun buildMatrix(alpha: Double, length: Int): DenseNDArray {

      val matrix = DenseNDArrayFactory.zeros(Shape(length, length))

      for (i in 0 until matrix.rows) {
        for (j in 0 until matrix.columns) {
          when {
            i == j -> matrix[i, j] = 1.0
            i >= j -> matrix[i, j] = alpha.pow(i - j)
            else -> matrix[i, j] = 0.0
          }
        }
      }

      return matrix
    }
  }

  /**
   * TODO: write documentation
   */
  override val propagateToInput: Boolean = true

  /**
   * TODO: write documentation
   */
  override val useDropout: Boolean = false

  /**
   * TODO: write documentation
   */
  private lateinit var matrix: DenseNDArray

  /**
   * TODO: write documentation
   */
  private lateinit var inputErrors: List<DenseNDArray>

  /**
   * The Forward.
   *
   * @param input the input
   *
   * @return the result of the forward
   */
  override fun forward(input: List<DenseNDArray>): List<DenseNDArray> {

    this.matrix = buildMatrix(this.alpha, input.size)

    return this.matrix.dot(input.toMatrix()).let { out ->
      (0 until out.rows).map { out.getRow(it).t }
    }
  }

  /**
   * The Backward.
   *
   * @param outputErrors the output errors
   */
  override fun backward(outputErrors: List<DenseNDArray>) {

    this.inputErrors = outputErrors.toMatrix().dot(this.matrix.t).let { out ->
      (0 until out.rows).map { out.getRow(it).t }
    }
  }

  /**
   * Return the input errors of the last backward.
   * Before calling this method make sure that [propagateToInput] is enabled.
   *
   * @param copy whether to return by value or by reference (default true)
   *
   * @return the input errors
   */
  override fun getInputErrors(copy: Boolean): List<DenseNDArray> = this.inputErrors.map { if (copy) it.copy() else it }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the network parameters
   */
  override fun getParamsErrors(copy: Boolean): ParamsErrorsList = emptyList()
}