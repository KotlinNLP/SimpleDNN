/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.simplemath.ndarray

import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator

/**
 *
 */
class SparseBinaryNDArray : NDArray<SparseBinaryNDArray> {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   *
   */
  override val factory = SparseBinaryNDArrayFactory

  /**
   *
   */
  override val isVector: Boolean
    get() = TODO("not implemented") //To change initializer of created properties use File | Settings | File Templates.

  /**
   *
   */
  override val isMatrix: Boolean
    get() = TODO("not implemented") //To change initializer of created properties use File | Settings | File Templates.

  /**
   *
   */
  override val length: Int
    get() = TODO("not implemented") //To change initializer of created properties use File | Settings | File Templates.

  /**
   *
   */
  override val rows: Int
    get() = TODO("not implemented") //To change initializer of created properties use File | Settings | File Templates.

  /**
   *
   */
  override val columns: Int
    get() = TODO("not implemented") //To change initializer of created properties use File | Settings | File Templates.

  /**
   *
   */
  override val shape: Shape
    get() = TODO("not implemented") //To change initializer of created properties use File | Settings | File Templates.

  /**
   *
   */
  override val isOneHotEncoder: Boolean
    get() = TODO("not implemented") //To change initializer of created properties use File | Settings | File Templates.

  /**
   * Transpose
   */
  override val T: SparseBinaryNDArray
    get() = TODO("not implemented") //To change initializer of created properties use File | Settings | File Templates.

  /**
   *
   */
  override fun get(i: Int): Double {
    TODO("not implemented")
  }

  /**
   * Get the i-th row
   *
   * @param i the index of the row to be returned
   *
   * @return the selected row as a new [SparseBinaryNDArray]
   */
  override fun getRow(i: Int): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   * Get the i-th column
   *
   * @param i the index of the column to be returned
   *
   * @return the selected column as a new [SparseBinaryNDArray]
   */
  override fun getColumn(i: Int): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun get(i: Int, j: Int): Double {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun set(i: Int, value: Double) {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun copy(): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun set(i: Int, j: Int, value: Double) {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignValues(a: NDArray<*>): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignValues(n: Double): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun zeros(): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun sum(): Double {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun avg(): Double {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun sum(n: Double): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   * @return the index of the maximum value (-1 if empty)
   */
  override fun argMaxIndex(): Int {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun sum(a: SparseBinaryNDArray): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignSum(a: NDArray<*>): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignSum(n: Double): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignSum(a: SparseBinaryNDArray, n: Double): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignSum(a: SparseBinaryNDArray, b: SparseBinaryNDArray): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   * Euclidean norm of this NDArray
   *
   * @return the euclidean norm
   */
  override fun norm2(): Double {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun equals(other: Any?): Boolean {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun hashCode(): Int {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun sub(n: Double): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun sub(a: SparseBinaryNDArray): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   * In-place subtraction by number
   */
  override fun assignSub(n: Double): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignSub(a: SparseBinaryNDArray): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun reverseSub(n: Double): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun dot(a: SparseBinaryNDArray): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   * Dot product between this [SparseBinaryNDArray] and a [DenseNDArray] masked by [mask]
   *
   * @param a the [DenseNDArray] by which is calculated the dot product
   * @param mask the mask applied to a
   *
   * @return a [SparseNDArray]
   */
  override fun dot(a: DenseNDArray, mask: NDArrayMask): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignDot(a: SparseBinaryNDArray, b: SparseBinaryNDArray): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignDot(a: DenseNDArray, b: NDArray<*>): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun prod(n: Double): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun prod(a: SparseBinaryNDArray): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignProd(n: Double): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignProd(a: SparseBinaryNDArray, n: Double): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignProd(a: SparseBinaryNDArray, b: SparseBinaryNDArray): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignProd(a: SparseBinaryNDArray): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun div(n: Double): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun div(a: SparseBinaryNDArray): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignDiv(n: Double): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignDiv(a: SparseBinaryNDArray): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   * Round values to Int
   *
   * @param threshold a value is rounded to the next Int if is >= [threshold], to the previous otherwise
   *
   * @return a new NDArray with the values of the current one rounded to Int
   */
  override fun roundInt(threshold: Double): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   * Round values to Int in-place
   *
   * @param threshold a value is rounded to the next Int if is >= [threshold], to the previous otherwise
   *
   * @return this [SparseBinaryNDArray]
   */
  override fun assignRoundInt(threshold: Double): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   * Sign function
   *
   * @return a new [SparseBinaryNDArray] containing the results of the function sign() applied element-wise
   */
  override fun sign(): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun randomize(randomGenerator: RandomGenerator): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun sqrt(): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   * Power
   *
   * @param power the exponent
   *
   * @return a new [SparseBinaryNDArray] containing the values of this to the power of [power]
   */
  override fun pow(power: Double): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   * In-place power
   *
   * @param power the exponent
   *
   * @return this [SparseBinaryNDArray] to the power of [power]
   */
  override fun assignPow(power: Double): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun equals(a: SparseBinaryNDArray, tolerance: Double): Boolean {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun zerosLike(): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun concatH(a: SparseBinaryNDArray): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun concatV(a: SparseBinaryNDArray): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   * Return a one-dimensional NDArray sub-vector of a vertical vector
   */
  override fun getRange(a: Int, b: Int): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun toString(): String {
    TODO("not implemented")
  }
}
