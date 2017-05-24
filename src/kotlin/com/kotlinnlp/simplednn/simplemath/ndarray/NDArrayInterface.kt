/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.simplemath.ndarray

import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import java.io.Serializable

interface NDArrayInterface: Serializable {

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
  operator fun get(i: Int): Number

  /**
   *
   */
  operator fun get(i: Int, j: Int): Number


  /**
   *
   */
  operator fun set(i: Int, value: Number)

  /**
   *
   */
  operator fun set(i: Int, j: Int, value: Number)

  /**
   * Get the i-th row
   *
   * @param i the index of the row to be returned
   *
   * @return the selected row as a new NDArrayInterface
   */
  fun getRow(i: Int): NDArrayInterface

  /**
   * Get the i-th column
   *
   * @param i the index of the column to be returned
   *
   * @return the selected column as a new NDArrayInterface
   */
  fun getColumn(i: Int): NDArrayInterface

  /**
   * transpose
   */
  val T: NDArrayInterface
    get

  /**
   *
   */
  fun copy(): NDArrayInterface

  /**
   *
   */
  fun zeros(): NDArrayInterface

  /**
   *
   */
  fun assignValues(a: NDArrayInterface): NDArrayInterface

  /**
   *
   */
  fun assignValues(n: Number): NDArrayInterface

  /**
   *
   */
  fun sum(n: Number): NDArrayInterface

  /**
   *
   */
  fun sum(a: NDArrayInterface): NDArrayInterface

  /**
   *
   */
  fun sum(): Double

  /**
   *
   */
  fun assignSum(n: Number): NDArrayInterface

  /**
   *
   */
  fun assignSum(a: NDArrayInterface, n: Number): NDArrayInterface

  /**
   *
   */
  fun assignSum(a: NDArrayInterface, b: NDArrayInterface): NDArrayInterface

  /**
   *
   */
  fun assignSum(a: NDArrayInterface): NDArrayInterface

  /**
   *
   */
  fun sub(n: Number): NDArrayInterface

  /**
   *
   */
  fun sub(a: NDArrayInterface): NDArrayInterface

  /**
   * In-place subtraction by number
   */
  fun assignSub(n: Number): NDArrayInterface

  /**
   *
   */
  fun assignSub(a: NDArrayInterface): NDArrayInterface

  /**
   *
   */
  fun reverseSub(n: Number): NDArrayInterface

  /**
   *
   */
  fun dot(a: NDArrayInterface): NDArrayInterface

  /**
   *
   */
  fun assignDot(a: NDArrayInterface, b: NDArrayInterface): NDArrayInterface

  /**
   *
   */
  fun prod(n: Number): NDArrayInterface

  /**
   *
   */
  fun prod(a: NDArrayInterface): NDArrayInterface

  /**
   *
   */
  fun assignProd(n: Number): NDArrayInterface

  /**
   *
   */
  fun assignProd(a: NDArrayInterface, n: Number): NDArrayInterface

  /**
   *
   */
  fun assignProd(a: NDArrayInterface, b: NDArrayInterface): NDArrayInterface

  /**
   *
   */
  fun assignProd(a: NDArrayInterface): NDArrayInterface

  /**
   *
   */
  fun div(n: Number): NDArrayInterface

  /**
   *
   */
  fun div(a: NDArrayInterface): NDArrayInterface

  /**
   *
   */
  fun assignDiv(n: Number): NDArrayInterface

  /**
   *
   */
  fun assignDiv(a: NDArrayInterface): NDArrayInterface

  /**
   * Round values to Int
   *
   * @param threshold a value is rounded to the next Int if is >= [threshold], to the previous otherwise
   *
   * @return a new NDArrayInterface with the values of the current one rounded to Int
   */
  fun roundInt(threshold: Double = 0.5): NDArrayInterface

  /**
   * Round values to Int in-place
   *
   * @param threshold a value is rounded to the next Int if is >= [threshold], to the previous otherwise
   *
   * @return this NDArrayInterface
   */
  fun assignRoundInt(threshold: Double = 0.5): NDArrayInterface

  /**
   *
   */
  fun avg(): Double

  /**
   * Sign function
   *
   * @return a new NDArrayInterface containing the results of the function sign() applied element-wise
   */
  fun sign(): NDArrayInterface

  /**
   * @return the index of the maximum value (-1 if empty)
   **/
  fun argMaxIndex(): Int

  /**
   *
   */
  fun randomize(randomGenerator: RandomGenerator): NDArrayInterface

  /**
   *
   */
  fun sqrt(): NDArrayInterface

  /**
   * Power
   *
   * @param power the exponent
   *
   * @return a new [NDArrayInterface] containing the values of this to the power of [power]
   */
  fun pow(power: Double): NDArrayInterface

  /**
   * In-place power
   *
   * @param power the exponent
   *
   * @return this [NDArrayInterface] to the power of [power]
   */
  fun assignPow(power: Double): NDArrayInterface

  /**
   * Euclidean norm of this NDArrayInterface
   *
   * @return the euclidean norm
   */
  fun norm2(): Double

  /**
   *
   */
  fun equals(a: NDArrayInterface, tolerance: Double = 10e-4): Boolean

  /**
   *
   */
  fun zerosLike(): NDArrayInterface

  /**
   *
   */
  fun concatH(a: NDArrayInterface): NDArrayInterface

  /**
   *
   */
  fun concatV(a: NDArrayInterface): NDArrayInterface

  /**
   * Return a one-dimensional NDArrayInterface sub-vector of a vertical vector
   */
  fun getRange(a: Int, b: Int): NDArrayInterface

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
