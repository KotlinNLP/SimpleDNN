/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.simplemath.ndarray

import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import java.io.Serializable

/**
 *
 */
interface NDArray : Serializable {

  /**
   *
   */
  val isVector: Boolean
    get

  /**
   *
   */
  val isMatrix: Boolean
    get

  /**
   *
   */
  val length: Int
    get

  /**
   *
   */
  val rows: Int
    get

  /**
   *
   */
  val columns: Int
    get

  /**
   *
   */
  val shape: Shape
    get

  /**
   *
   */
  val isOneHotEncoder: Boolean

  /**
   *
   */
  operator fun get(i: Int): Double

  /**
   *
   */
  operator fun get(i: Int, j: Int): Double


  /**
   *
   */
  operator fun set(i: Int, value: Double)

  /**
   *
   */
  operator fun set(i: Int, j: Int, value: Double)

  /**
   * Get the i-th row
   *
   * @param i the index of the row to be returned
   *
   * @return the selected row as a new NDArray
   */
  fun getRow(i: Int): NDArray

  /**
   * Get the i-th column
   *
   * @param i the index of the column to be returned
   *
   * @return the selected column as a new NDArray
   */
  fun getColumn(i: Int): NDArray

  /**
   * transpose
   */
  val T: NDArray
    get

  /**
   *
   */
  fun copy(): NDArray

  /**
   *
   */
  fun zeros(): NDArray

  /**
   *
   */
  fun assignValues(a: NDArray): NDArray

  /**
   *
   */
  fun assignValues(n: Double): NDArray

  /**
   *
   */
  fun sum(n: Double): NDArray

  /**
   *
   */
  fun sum(a: NDArray): NDArray

  /**
   *
   */
  fun sum(): Double

  /**
   *
   */
  fun assignSum(n: Double): NDArray

  /**
   *
   */
  fun assignSum(a: NDArray, n: Double): NDArray

  /**
   *
   */
  fun assignSum(a: NDArray, b: NDArray): NDArray

  /**
   *
   */
  fun assignSum(a: NDArray): NDArray

  /**
   *
   */
  fun sub(n: Double): NDArray

  /**
   *
   */
  fun sub(a: NDArray): NDArray

  /**
   * In-place subtraction by number
   */
  fun assignSub(n: Double): NDArray

  /**
   *
   */
  fun assignSub(a: NDArray): NDArray

  /**
   *
   */
  fun reverseSub(n: Double): NDArray

  /**
   *
   */
  fun dot(a: NDArray): NDArray

  /**
   *
   */
  fun assignDot(a: NDArray, b: NDArray): NDArray

  /**
   *
   */
  fun prod(n: Double): NDArray

  /**
   *
   */
  fun prod(a: NDArray): NDArray

  /**
   *
   */
  fun assignProd(n: Double): NDArray

  /**
   *
   */
  fun assignProd(a: NDArray, n: Double): NDArray

  /**
   *
   */
  fun assignProd(a: NDArray, b: NDArray): NDArray

  /**
   *
   */
  fun assignProd(a: NDArray): NDArray

  /**
   *
   */
  fun div(n: Double): NDArray

  /**
   *
   */
  fun div(a: NDArray): NDArray

  /**
   *
   */
  fun assignDiv(n: Double): NDArray

  /**
   *
   */
  fun assignDiv(a: NDArray): NDArray

  /**
   * Round values to Int
   *
   * @param threshold a value is rounded to the next Int if is >= [threshold], to the previous otherwise
   *
   * @return a new NDArray with the values of the current one rounded to Int
   */
  fun roundInt(threshold: Double = 0.5): NDArray

  /**
   * Round values to Int in-place
   *
   * @param threshold a value is rounded to the next Int if is >= [threshold], to the previous otherwise
   *
   * @return this NDArray
   */
  fun assignRoundInt(threshold: Double = 0.5): NDArray

  /**
   *
   */
  fun avg(): Double

  /**
   * Sign function
   *
   * @return a new NDArray containing the results of the function sign() applied element-wise
   */
  fun sign(): NDArray

  /**
   * @return the index of the maximum value (-1 if empty)
   **/
  fun argMaxIndex(): Int

  /**
   *
   */
  fun randomize(randomGenerator: RandomGenerator): NDArray

  /**
   *
   */
  fun sqrt(): NDArray

  /**
   * Power
   *
   * @param power the exponent
   *
   * @return a new [NDArray] containing the values of this to the power of [power]
   */
  fun pow(power: Double): NDArray

  /**
   * In-place power
   *
   * @param power the exponent
   *
   * @return this [NDArray] to the power of [power]
   */
  fun assignPow(power: Double): NDArray

  /**
   * Euclidean norm of this NDArray
   *
   * @return the euclidean norm
   */
  fun norm2(): Double

  /**
   *
   */
  fun equals(a: DenseNDArray, tolerance: Double = 10e-4): Boolean

  /**
   *
   */
  fun zerosLike(): NDArray

  /**
   *
   */
  fun concatH(a: NDArray): NDArray

  /**
   *
   */
  fun concatV(a: NDArray): NDArray

  /**
   * Return a one-dimensional NDArray sub-vector of a vertical vector
   */
  fun getRange(a: Int, b: Int): NDArray

  /**
   *
   */
  override fun toString(): String

  /**
   *
   */
  override fun equals(other: Any?): Boolean

  /**
   *
   */
  override fun hashCode(): Int
}
