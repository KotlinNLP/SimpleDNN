/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.simplemath.ndarray

import org.jblas.DoubleMatrix

/**
 *
 */
object NDArrayFactory {

  /**
   * @param shape shape
   * @return a new empty DenseNDArray
   */
  fun emptyArray(shape: Shape): NDArray {
    return DenseNDArray(DoubleMatrix.zeros(shape.dim1, shape.dim2))
  }

  /**
   *
   */
  fun arrayOf(vector: DoubleArray): NDArray {
    val m = DoubleMatrix(vector.size, 1)

    (0 until vector.size).forEach { i -> m.put(i, vector[i]) }

    return DenseNDArray(m)
  }

  /**
   *
   */
  fun arrayOf(matrix: Array<DoubleArray>): NDArray {
    val rows = matrix.size
    val columns = matrix[0].size
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
   *
   * @param shape shape
   * @return a new DenseNDArray filled with zeros
   */
  fun zeros(shape: Shape): NDArray {
    return this.emptyArray(shape)
  }

  /**
   * Build a new DenseNDArray filled with zeros but one with 1.0
   *
   * @param length the length of the array
   * @param oneAt the index of the one element
   * @return a oneHotEncoder DenseNDArray
   */
  fun oneHotEncoder(length: Int, oneAt: Int): NDArray {
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
  fun random(shape: Shape, from: Double, to: Double): NDArray {

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
}
