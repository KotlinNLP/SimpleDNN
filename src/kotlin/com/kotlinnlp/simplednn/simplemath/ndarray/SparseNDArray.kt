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
class SparseNDArray : NDArray<SparseNDArray> {

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
  override val factory = SparseNDArrayFactory

  /**
   *
   */
  override val isVector: Boolean
    get() = TODO("not implemented")

  /**
   *
   */
  override val isMatrix: Boolean
    get() = TODO("not implemented")

  /**
   *
   */
  override val length: Int
    get() = TODO("not implemented")

  /**
   *
   */
  override val rows: Int
    get() = TODO("not implemented")

  /**
   *
   */
  override val columns: Int
    get() = TODO("not implemented")

  /**
   *
   */
  override val shape: Shape
    get() = TODO("not implemented")

  /**
   *
   */
  override val isOneHotEncoder: Boolean
    get() = TODO("not implemented")

  /**
   * Transpose
   */
  override val T: SparseNDArray
    get() = TODO("not implemented")

  /**
   *
   */
  val mask: NDArrayMask
    get() = TODO("not implemented")

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
   * @return the selected row as a new [SparseNDArray]
   */
  override fun getRow(i: Int): SparseNDArray {
    TODO("not implemented")
  }

  /**
   * Get the i-th column
   *
   * @param i the index of the column to be returned
   *
   * @return the selected column as a new [SparseNDArray]
   */
  override fun getColumn(i: Int): SparseNDArray {
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
  override fun copy(): SparseNDArray {
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
  override fun assignValues(a: NDArray<*>): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignValues(n: Double): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun zeros(): SparseNDArray {
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
  override fun sum(n: Double): SparseNDArray {
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
  override fun sum(a: SparseNDArray): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignSum(n: Double): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignSum(a: NDArray<*>): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignSum(a: SparseNDArray, n: Double): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignSum(a: SparseNDArray, b: SparseNDArray): SparseNDArray {
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
  override fun toString(): String {
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
  override fun sub(n: Double): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun sub(a: SparseNDArray): SparseNDArray {
    TODO("not implemented")
  }

  /**
   * In-place subtraction by number
   */
  override fun assignSub(n: Double): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignSub(a: NDArray<*>): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun reverseSub(n: Double): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun dot(a: SparseNDArray): SparseNDArray {
    TODO("not implemented")
  }

  /**
   * Dot product between this [SparseNDArray] and a [DenseNDArray] masked by [mask]
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
  override fun assignDot(a: SparseNDArray, b: SparseNDArray): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignDot(a: DenseNDArray, b: NDArray<*>): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun prod(n: Double): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun prod(a: SparseNDArray): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignProd(n: Double): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignProd(n: Double, mask: NDArrayMask): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignProd(a: SparseNDArray, n: Double): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignProd(a: SparseNDArray, b: SparseNDArray): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignProd(a: SparseNDArray): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun div(n: Double): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun div(a: NDArray<*>): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignDiv(n: Double): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignDiv(a: SparseNDArray): SparseNDArray {
    TODO("not implemented")
  }

  /**
   * Round values to Int
   *
   * @param threshold a value is rounded to the next Int if is >= [threshold], to the previous otherwise
   *
   * @return a new NDArray with the values of the current one rounded to Int
   */
  override fun roundInt(threshold: Double): SparseNDArray {
    TODO("not implemented")
  }

  /**
   * Round values to Int in-place
   *
   * @param threshold a value is rounded to the next Int if is >= [threshold], to the previous otherwise
   *
   * @return this [SparseNDArray]
   */
  override fun assignRoundInt(threshold: Double): SparseNDArray {
    TODO("not implemented")
  }

  /**
   * Sign function
   *
   * @return a new [SparseNDArray] containing the results of the function sign() applied element-wise
   */
  override fun sign(): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun randomize(randomGenerator: RandomGenerator): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun sqrt(): SparseNDArray {
    TODO("not implemented")
  }

  /**
   * Square root of this [SparseNDArray] masked by [mask]
   *
   * @param mask the mask to apply
   *
   * @return a [SparseNDArray]
   */
  override fun sqrt(mask: NDArrayMask): SparseNDArray {
    TODO("not implemented")
  }

  /**
   * Power
   *
   * @param power the exponent
   *
   * @return a new [SparseNDArray] containing the values of this to the power of [power]
   */
  override fun pow(power: Double): SparseNDArray {
    TODO("not implemented")
  }

  /**
   * In-place power
   *
   * @param power the exponent
   *
   * @return this [SparseNDArray] to the power of [power]
   */
  override fun assignPow(power: Double): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun equals(a: SparseNDArray, tolerance: Double): Boolean {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun zerosLike(): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun concatH(a: SparseNDArray): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun concatV(a: SparseNDArray): SparseNDArray {
    TODO("not implemented")
  }

  /**
   * Return a one-dimensional NDArray sub-vector of a vertical vector
   */
  override fun getRange(a: Int, b: Int): SparseNDArray {
    TODO("not implemented")
  }
}
