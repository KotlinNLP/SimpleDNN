/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.simplemath.ndarray.dense

import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray
import org.jblas.DoubleMatrix

/**
 *
 */
object DenseNDArrayFactory : NDArrayFactory<DenseNDArray> {

  /**
   * Private val used to serialize the class (needed by Serializable).
   */
  @Suppress("unused")
  private const val serialVersionUID: Long = 1L

  /**
   * @param shape shape
   * @return a new empty DenseNDArray
   */
  override fun emptyArray(shape: Shape): DenseNDArray {
    return DenseNDArray(DoubleMatrix.zeros(shape.dim1, shape.dim2))
  }

  /**
   *
   * @param shape shape
   * @return a new DenseNDArray filled with zeros
   */
  override fun zeros(shape: Shape): DenseNDArray {
    return this.emptyArray(shape)
  }

  /**
   * @param shape shape
   * @param value the init value
   *
   * @return a new [DenseNDArray] filled with the given value
   */
  override fun fill(shape: Shape, value: Double): DenseNDArray {
    return DenseNDArray(DoubleMatrix.zeros(shape.dim1, shape.dim2).fill(value))
  }

  /**
   * Build a new DenseNDArray filled with zeros but one with 1.0
   *
   * @param length the length of the array
   * @param oneAt the index of the one element
   * @return a oneHotEncoder DenseNDArray
   */
  override fun oneHotEncoder(length: Int, oneAt: Int): DenseNDArray {
    require(oneAt in 0 until length)

    val array = this.emptyArray(Shape(length))

    array[oneAt] = 1.0

    return array
  }

  /**
   * Build a new DenseNDArray filled with random values uniformly distributed in range [[from], [to]]
   *
   * @param shape shape
   * @param from inclusive lower bound of random values range
   * @param to inclusive upper bound of random values range
   * @return a new DenseNDArray filled with random values
   */
  override fun random(shape: Shape, from: Double, to: Double): DenseNDArray {

    val m = DoubleMatrix.rand(shape.dim1, shape.dim2)
    val rangeSize = to - from

    if (rangeSize != 1.0) {
      m.muli(rangeSize)
    }

    if (from != 0.0) {
      m.addi(from)
    }

    return DenseNDArray(m)
  }

  /**
   * @param shape shape
   *
   * @return a new DenseNDArray filled with ones
   */
  fun ones(shape: Shape): DenseNDArray {
    return DenseNDArray(DoubleMatrix.ones(shape.dim1, shape.dim2))
  }

  /**
   *
   */
  fun arrayOf(vector: DoubleArray): DenseNDArray {
    val m = DoubleMatrix(vector.size, 1)

    (0 until vector.size).forEach { i -> m.put(i, vector[i]) }

    return DenseNDArray(m)
  }

  /**
   *
   */
  fun arrayOf(matrix: List<DoubleArray>): DenseNDArray {
    val rows = matrix.size
    val columns = if (matrix.isNotEmpty()) matrix[0].size else 0
    val m = DoubleMatrix(rows, columns)

    (0 until rows * columns).forEach { linearIndex ->
      // linear indexing: loop rows before, column by column
      val row = linearIndex % rows
      val column = linearIndex / rows
      m.put(linearIndex, matrix[row][column])
    }

    return DenseNDArray(m)
  }

  /**
   * Build a [DenseNDArray] that contains the same values of a given generic NDArray.
   *
   * @param array a generic NDArray
   *
   * @return a new Dense NDArray
   */
  fun fromNDArray(array: NDArray<*>): DenseNDArray = when (array) {

    is DenseNDArray -> array.copy()

    is SparseNDArray -> {
      val ret: DenseNDArray = DenseNDArrayFactory.zeros(array.shape)
      array.forEach { (indices, value) -> ret[indices.first, indices.second] = value }
      ret
    }

    is SparseBinaryNDArray -> {
      val ret: DenseNDArray = DenseNDArrayFactory.zeros(array.shape)
      array.forEach { (i, j) -> ret[i, j] = 1.0 }
      ret
    }

    else -> throw RuntimeException("Invalid NDArray type.")
  }
}
