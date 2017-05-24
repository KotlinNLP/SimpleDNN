/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.simplemath.ndarray

import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.simplemath.equals
import org.jblas.DoubleMatrix
import org.jblas.DoubleMatrix.concatHorizontally
import org.jblas.DoubleMatrix.concatVertically
import org.jblas.MatrixFunctions

/**
 * NDArray with dense values (implemented using JBlas)
 */
class DenseNDArray(private val storage: DoubleMatrix) : NDArray {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * Whether the array is a row or a column vector
   */
  override val isVector: Boolean
    get() = this.storage.rows == 1 || this.storage.columns == 1

  /**
   * Whether the array is a bi-dimensional matrix
   */
  override val isMatrix: Boolean
    get() = !this.isVector

  /**
   *
   */
  override val length: Int
    get() = this.storage.length

  /**
   *
   */
  override val rows: Int
    get() = this.storage.rows

  /**
   *
   */
  override val columns: Int
    get() = this.storage.columns

  /**
   *
   * @return
   */
  override val shape: Shape
    get() = Shape(this.rows, this.columns)

  /**
   *
   */
  override val isOneHotEncoder: Boolean get() {

    var isTrue = false

    if (this.isVector) {
      (0 until this.length)
        .filter { this[it] != 0.0 }
        .forEach {
          if (this[it] == 1.0 && !isTrue) {
            isTrue = true
          } else {
            return false
          }
        }
    }

    return isTrue
  }

  /**
   *
   */
  override operator fun get(i: Int): Double = this.storage.get(i)

  /**
   *
   */
  override operator fun get(i: Int, j: Int): Double = this.storage.get(i, j)

  /**
   *
   */
  override operator fun set(i: Int, value: Double) { this.storage.put(i, value) }

  /**
   *
   */
  override operator fun set(i: Int, j: Int, value: Double) { this.storage.put(i, j, value) }

  /**
   * Get the i-th row
   *
   * @param i the index of the row to be returned
   *
   * @return the selected row as a new DenseNDArray
   */
  override fun getRow(i: Int): NDArray {
    val values = this.storage.getRow(i)
    return NDArrayFactory.arrayOf(arrayOf<DoubleArray>(values.toArray()))
  }

  /**
   * Get the i-th column
   *
   * @param i the index of the column to be returned
   *
   * @return the selected column as a new DenseNDArray
   */
  override fun getColumn(i: Int): NDArray {
    return DenseNDArray(this.storage.getColumn(i))
  }

  /**
   *
   */
  override val T: NDArray
    get() = DenseNDArray(this.storage.transpose())

  /**
   *
   */
  override fun copy(): NDArray = DenseNDArray(this.storage.dup())

  /**
   *
   */
  override fun zeros(): NDArray {
    this.storage.fill(0.0)
    return this
  }

  /**
   *
   */
  override fun assignValues(n: Double): NDArray {
    this.storage.fill(n)
    return this
  }

  /**
   * Assign the values of a to this DenseNDArray (it works also among rows and columns vectors)
   */
  override fun assignValues(a: NDArray): NDArray { (a as DenseNDArray)
    require(this.shape == a.shape ||
      (this.isVector && a.isVector && this.length == a.length))

    System.arraycopy(a.storage.data, 0, this.storage.data, 0, this.length)

    return this
  }

  /**
   *
   */
  override fun sum(n: Double): NDArray {
    return DenseNDArray(this.storage.add(n))
  }

  /**
   *
   */
  override fun sum(a: NDArray): NDArray { (a as DenseNDArray)
    return DenseNDArray(this.storage.add(a.storage))
  }

  /**
   *
   */
  override fun sum(): Double = this.storage.sum()

  /**
   *
   */
  override fun assignSum(n: Double): NDArray {
    this.storage.addi(n)
    return this
  }

  /**
   *
   */
  override fun assignSum(a: NDArray, n: Double): NDArray { (a as DenseNDArray)
    a.storage.addi(n, this.storage)
    return this
  }

  /**
   * Assign a + b to this DenseNDArray (it works also among rows and columns vectors)
   */
  override fun assignSum(a: NDArray, b: NDArray): NDArray { (a as DenseNDArray); (b as DenseNDArray)
    a.storage.addi(b.storage, this.storage)
    return this
  }

  /**
   * Assign a to this DenseNDArray (it works also among rows and columns vectors)
   */
  override fun assignSum(a: NDArray): NDArray {(a as DenseNDArray)
    this.storage.addi(a.storage)
    return this
  }

  /**
   *
   */
  override fun sub(n: Double): NDArray {
    return DenseNDArray(this.storage.sub(n))
  }

  /**
   *
   */
  override fun sub(a: NDArray): NDArray { (a as DenseNDArray)
    return DenseNDArray(this.storage.sub(a.storage))
  }

  /**
   * In-place subtraction by number
   */
  override fun assignSub(n: Double): NDArray {
    this.storage.subi(n)
    return this
  }

  /**
   *
   */
  override fun assignSub(a: NDArray): NDArray { (a as DenseNDArray)
    this.storage.subi(a.storage)
    return this
  }

  /**
   *
   */
  override fun reverseSub(n: Double): NDArray {
    return DenseNDArray(this.storage.rsub(n))
  }

  /**
   *
   */
  override fun dot(a: NDArray): NDArray { (a as DenseNDArray)
    return DenseNDArray(this.storage.mmul(a.storage))
  }

  /**
   *
   */
  override fun assignDot(a: NDArray, b: NDArray): NDArray { (a as DenseNDArray); (b as DenseNDArray)
    require(a.rows == this.rows && b.columns == this.columns)
    a.storage.mmuli(b.storage, this.storage)
    return this
  }

  /**
   *
   */
  override fun prod(n: Double): NDArray {
    return DenseNDArray(this.storage.mul(n))
  }

  /**
   *
   */
  override fun prod(a: NDArray): NDArray { (a as DenseNDArray)
    return DenseNDArray(this.storage.mul(a.storage))
  }

  /**
   *
   */
  override fun assignProd(a: NDArray, n: Double): NDArray { (a as DenseNDArray)
    a.storage.muli(n, this.storage)
    return this
  }

  /**
   *
   */
  override fun assignProd(a: NDArray, b: NDArray): NDArray { (a as DenseNDArray); (b as DenseNDArray)
    a.storage.muli(b.storage, this.storage)
    return this
  }

  /**
   *
   */
  override fun assignProd(a: NDArray): NDArray { (a as DenseNDArray)
    this.storage.muli(a.storage)
    return this
  }

  /**
   *
   */
  override fun assignProd(n: Double): NDArray {
    this.storage.muli(n)
    return this
  }

  /**
   *
   */
  override fun div(n: Double): NDArray {
    return DenseNDArray(this.storage.div(n))
  }

  /**
   *
   */
  override fun div(a: NDArray): NDArray { (a as DenseNDArray)
    return DenseNDArray(this.storage.div(a.storage))
  }

  /**
   *
   */
  override fun assignDiv(n: Double): NDArray {
    this.storage.divi(n)
    return this
  }

  /**
   *
   */
  override fun assignDiv(a: NDArray): NDArray { (a as DenseNDArray)
    this.storage.divi(a.storage)
    return this
  }

  /**
   * Round values to Int
   *
   * @param threshold a value is rounded to the next Int if is >= [threshold], to the previous otherwise
   *
   * @return a new DenseNDArray with the values of the current one rounded to Int
   */
  override fun roundInt(threshold: Double): NDArray {

    val out = NDArrayFactory.emptyArray(this.shape)
    val floorValues = MatrixFunctions.floor(this.storage)

    for (i in 0 until this.length) {
      out[i] = if (this.storage[i] < threshold) floorValues[i] else floorValues[i] + 1
    }

    return out
  }

  /**
   * Round values to Int in-place
   *
   * @param threshold a value is rounded to the next Int if is >= [threshold], to the previous otherwise
   *
   * @return this DenseNDArray
   */
  override fun assignRoundInt(threshold: Double): NDArray {

    val floorValues = MatrixFunctions.floor(this.storage)

    for (i in 0 until this.length) {
      this[i] = if (this.storage[i] < threshold) floorValues[i] else floorValues[i] + 1
    }

    return this
  }

  /**
   *
   */
  override fun avg(): Double = this.storage.mean()

  /**
   * Sign function
   *
   * @return a new DenseNDArray containing the results of the function sign() applied element-wise
   */
  override fun sign(): NDArray {
    return DenseNDArray(MatrixFunctions.signum(this.storage))
  }

  /**
   * @return the index of the maximum value (-1 if empty)
   **/
  override fun argMaxIndex(): Int {

    var maxIndex: Int = -1
    var maxValue: Double? = null

    (0 until this.length).forEach { i ->
      val value = this[i]

      if (maxValue == null || value > maxValue!!) {
        maxValue = value
        maxIndex = i
      }
    }

    return maxIndex
  }

  /**
   *
   */
  override fun randomize(randomGenerator: RandomGenerator): NDArray {
    for (i in 0 until this.length) this[i] = randomGenerator.next() // i: linear index
    return this
  }

  /**
   *
   */
  override fun sqrt(): NDArray {
    return DenseNDArray(MatrixFunctions.sqrt(this.storage))
  }


  /**
   * Power
   *
   * @param power the exponent
   *
   * @return a new [DenseNDArray] containing the values of this to the power of [power]
   */
  override fun pow(power: Double): NDArray {
    return DenseNDArray(MatrixFunctions.pow(this.storage, power))
  }

  /**
   * In-place power
   *
   * @param power the exponent
   *
   * @return this [DenseNDArray] to the power of [power]
   */
  override fun assignPow(power: Double): NDArray {
    MatrixFunctions.powi(this.storage, power)
    return this
  }

  /**
   * Euclidean norm of this DenseNDArray
   *
   * @return the euclidean norm
   */
  override fun norm2(): Double {
    val zeros = this.zerosLike() as DenseNDArray
    return this.storage.distance2(zeros.storage)
  }

  /**
   *
   */
  override fun concatH(a: NDArray): NDArray { (a as DenseNDArray)
    return DenseNDArray(concatHorizontally(this.storage, a.storage))
  }

  /**
   *
   */
  override fun concatV(a: NDArray): NDArray { (a as DenseNDArray)
    return DenseNDArray(concatVertically(this.storage, a.storage))
  }

  /**
   * Return a one-dimensional DenseNDArray sub-vector of a vertical vector
   */
  override fun getRange(a: Int, b: Int): NDArray {
    require(this.shape.dim2 == 1)
    return DenseNDArray(this.storage.getRange(a, b))
  }

  /**
   *
   */
  override fun zerosLike(): NDArray {
    return DenseNDArray(DoubleMatrix.zeros(this.shape.dim1, shape.dim2))
  }

  /**
   * @param a a DenseNDArray
   * @param tolerance a must be in the range [a - tolerance, a + tolerance] to return True
   *
   * @return a Boolean which indicates if a is equal to be within the tolerance
   */
  override fun equals(a: DenseNDArray, tolerance: Double): Boolean {
    require(this.shape == a.shape)

    return (0 until this.length).all { equals(this[it], a[it], tolerance) }
  }

  /**
   *
   */
  override fun toString(): String = this.storage.toString()

  /**
   *
   */
  override fun equals(other: Any?): Boolean {
    return other is DenseNDArray && this.equals(other)
  }

  /**
   *
   */
  override fun hashCode(): Int {
    return this.storage.hashCode()
  }
}
