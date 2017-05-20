/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.simplemath.ndarray.wrapper

import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.simplemath.equals
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArrayInterface
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import org.jblas.DoubleMatrix
import org.jblas.DoubleMatrix.concatHorizontally
import org.jblas.DoubleMatrix.concatVertically
import org.jblas.MatrixFunctions

/**
 * NDArrayInterface implementation using JBlas
 */
class JBlasArray(private val storage: DoubleMatrix) : NDArrayInterface {

  companion object: NDArrayFactory {

    /**
     * @param shape shape
     * @return a new empty JBlasArray
     */
    override fun emptyArray(shape: Shape): JBlasArray {
      return JBlasArray(DoubleMatrix.zeros(shape.dim1, shape.dim2))
    }

    /**
     *
     */
    override fun arrayOf(vector: DoubleArray): JBlasArray {
      val m = DoubleMatrix(vector.size, 1)

      (0 until vector.size).forEach { i -> m.put(i, vector[i]) }

      return JBlasArray(m)
    }

    /**
     *
     */
    override fun arrayOf(matrix: Array<DoubleArray>): JBlasArray {
      val rows = matrix.size
      val columns = matrix[0].size
      val m = DoubleMatrix(rows, columns)

      (0 until rows * columns).forEach { linearIndex ->
        // linear indexing: loop rows before, column by column
        val row = linearIndex % rows
        val column = linearIndex / rows
        m.put(linearIndex, matrix[row][column])
      }

      return JBlasArray(m)
    }

    /**
     *
     * @param shape shape
     * @return a new JBlasArray filled with zeros
     */
    override fun zeros(shape: Shape): JBlasArray {
      return emptyArray(shape)
    }

    /**
     * Build a new JBlasArray filled with zeros but one with 1.0
     *
     * @param length the length of the array
     * @param oneAt the index of the one element
     * @return a oneHotEncoder JBlasArray
     */
    override fun oneHotEncoder(length: Int, oneAt: Int): JBlasArray {
      require(oneAt in 0 until length)

      val array = emptyArray(Shape(length))

      array[oneAt] = 1.0

      return array
    }

    /**
   * Build a new JBlasArray filled with random values uniformly distributed in range [[from], [to]]
   *
   * @param shape shape
   * @param from inclusive lower bound of random values range
   * @param to inclusive upper bound of random values range
   * @return a new JBlasArray filled with random values
     */
    override fun random(shape: Shape, from: Double, to: Double): JBlasArray {

      val m = DoubleMatrix.rand(shape.dim1, shape.dim2)
      val rangeSize = to - from

      if (rangeSize != 1.0) {
        m.muli(rangeSize)
      }

      if (from != 0.0) {
        m.addi(from)
      }

      return JBlasArray(m)
    }
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
        .filter { this[it].toDouble() != 0.0 }
        .forEach {
          if (this[it].toDouble() == 1.0 && !isTrue) {
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
  override operator fun get(i: Int): Number = this.storage.get(i)

  /**
   *
   */
  override operator fun get(i: Int, j: Int): Number = this.storage.get(i, j)

  /**
   *
   */
  override operator fun set(i: Int, value: Number) { this.storage.put(i, value.toDouble()) }

  /**
   *
   */
  override operator fun set(i: Int, j: Int, value: Number) { this.storage.put(i, j, value.toDouble()) }

  /**
   * Get the i-th row
   *
   * @param i the index of the row to be returned
   *
   * @return the selected row as a new JBlasArray
   */
  override fun getRow(i: Int): JBlasArray {
    val values = this.storage.getRow(i)
    return JBlasArray.arrayOf(arrayOf<DoubleArray>(values.toArray()))
  }

  /**
   * Get the i-th column
   *
   * @param i the index of the column to be returned
   *
   * @return the selected column as a new JBlasArray
   */
  override fun getColumn(i: Int): JBlasArray {
    return JBlasArray(this.storage.getColumn(i))
  }

  /**
   *
   */
  override val T: JBlasArray
    get() = JBlasArray(this.storage.transpose())

  /**
   *
   */
  override fun copy(): JBlasArray = JBlasArray(this.storage.dup())

  /**
   *
   */
  override fun zeros(): JBlasArray {
    this.storage.fill(0.0)
    return this
  }

  /**
   *
   */
  override fun assignValues(n: Number): JBlasArray {
    this.storage.fill(n.toDouble())
    return this
  }

  /**
   * Assign the values of a to this JBlasArray (it works also among rows and columns vectors)
   */
  override fun assignValues(a: NDArrayInterface): JBlasArray { (a as JBlasArray)
    require(this.shape == a.shape ||
      (this.isVector && a.isVector && this.length == a.length))

    System.arraycopy(a.storage.data, 0, this.storage.data, 0, this.length)

    return this
  }

  /**
   *
   */
  override fun sum(n: Number): JBlasArray {
    return JBlasArray(this.storage.add(n.toDouble()))
  }

  /**
   *
   */
  override fun sum(a: NDArrayInterface): JBlasArray { (a as JBlasArray)
    return JBlasArray(this.storage.add(a.storage))
  }

  /**
   *
   */
  override fun sum(): Double = this.storage.sum()

  /**
   *
   */
  override fun assignSum(a: NDArrayInterface, n: Number): JBlasArray { (a as JBlasArray)
    a.storage.addi(n.toDouble(), this.storage)
    return this
  }

  /**
   * Assign a + b to this JBlasArray (it works also among rows and columns vectors)
   */
  override fun assignSum(a: NDArrayInterface, b: NDArrayInterface): JBlasArray { (a as JBlasArray); (b as JBlasArray)
    a.storage.addi(b.storage, this.storage)
    return this
  }

  /**
   * Assign a to this JBlasArray (it works also among rows and columns vectors)
   */
  override fun assignSum(a: NDArrayInterface): JBlasArray {(a as JBlasArray)
    this.storage.addi(a.storage)
    return this
  }

  /**
   *
   */
  override fun sub(n: Number): JBlasArray {
    return JBlasArray(this.storage.sub(n.toDouble()))
  }

  /**
   *
   */
  override fun sub(a: NDArrayInterface): JBlasArray { (a as JBlasArray)
    return JBlasArray(this.storage.sub(a.storage))
  }

  /**
   * In-place subtraction by number
   */
  override fun assignSub(n: Number): JBlasArray {
    this.storage.subi(n.toDouble())
    return this
  }

  /**
   *
   */
  override fun assignSub(a: NDArrayInterface): JBlasArray { (a as JBlasArray)
    this.storage.subi(a.storage)
    return this
  }

  /**
   *
   */
  override fun reverseSub(n: Number): JBlasArray {
    return JBlasArray(this.storage.rsub(n.toDouble()))
  }

  /**
   *
   */
  override fun dot(a: NDArrayInterface): JBlasArray { (a as JBlasArray)
    return JBlasArray(this.storage.mmul(a.storage))
  }

  /**
   *
   */
  override fun assignDot(a: NDArrayInterface, b: NDArrayInterface): JBlasArray { (a as JBlasArray); (b as JBlasArray)
    require(a.rows == this.rows && b.columns == this.columns)
    a.storage.mmuli(b.storage, this.storage)
    return this
  }

  /**
   *
   */
  override fun prod(n: Number): JBlasArray {
    return JBlasArray(this.storage.mul(n.toDouble()))
  }

  /**
   *
   */
  override fun prod(a: NDArrayInterface): JBlasArray { (a as JBlasArray)
    return JBlasArray(this.storage.mul(a.storage))
  }

  /**
   *
   */
  override fun assignProd(a: NDArrayInterface, n: Number): JBlasArray { (a as JBlasArray)
    a.storage.muli(n.toDouble(), this.storage)
    return this
  }

  /**
   *
   */
  override fun assignProd(a: NDArrayInterface, b: NDArrayInterface): JBlasArray { (a as JBlasArray); (b as JBlasArray)
    a.storage.muli(b.storage, this.storage)
    return this
  }

  /**
   *
   */
  override fun assignProd(a: NDArrayInterface): JBlasArray { (a as JBlasArray)
    this.storage.muli(a.storage)
    return this
  }

  /**
   *
   */
  override fun assignProd(n: Number): JBlasArray {
    this.storage.muli(n.toDouble())
    return this
  }

  /**
   *
   */
  override fun div(n: Number): JBlasArray {
    return JBlasArray(this.storage.div(n.toDouble()))
  }

  /**
   *
   */
  override fun div(a: NDArrayInterface): JBlasArray { (a as JBlasArray)
    return JBlasArray(this.storage.div(a.storage))
  }

  /**
   *
   */
  override fun assignDiv(n: Number): JBlasArray {
    this.storage.divi(n.toDouble())
    return this
  }

  /**
   *
   */
  override fun assignDiv(a: NDArrayInterface): JBlasArray { (a as JBlasArray)
    this.storage.divi(a.storage)
    return this
  }

  /**
   * Round values to Int
   *
   * @param threshold a value is rounded to the next Int if is >= [threshold], to the previous otherwise
   *
   * @return a new JBlasArray with the values of the current one rounded to Int
   */
  override fun roundInt(threshold: Double): JBlasArray {

    val out = emptyArray(this.shape)
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
   * @return this JBlasArray
   */
  override fun assignRoundInt(threshold: Double): JBlasArray {

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
   * @return a new JBlasArray containing the results of the function sign() applied element-wise
   */
  override fun sign(): JBlasArray {
    return JBlasArray(MatrixFunctions.signum(this.storage))
  }

  /**
   * @return the index of the maximum value (-1 if empty)
   **/
  override fun argMaxIndex(): Int {

    var maxIndex: Int = -1
    var maxValue: Double? = null

    (0 until this.length).forEach { i ->
      val value = this[i].toDouble()

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
  override fun randomize(randomGenerator: RandomGenerator): JBlasArray {
    for (i in 0 until this.length) this[i] = randomGenerator.next() // i: linear index
    return this
  }

  /**
   *
   */
  override fun sqrt(): JBlasArray {
    return JBlasArray(MatrixFunctions.sqrt(this.storage))
  }

  /**
   *
   */
  override fun pow(power: Double): JBlasArray {
    return JBlasArray(MatrixFunctions.pow(this.storage, power))
  }

  /**
   * Euclidean norm of this JBlasArray
   *
   * @return the euclidean norm
   */
  override fun norm2(): Double {
    return this.storage.distance2(this.zerosLike().storage)
  }

  /**
   *
   */
  override fun concatH(a: NDArrayInterface): JBlasArray { (a as JBlasArray)
    return JBlasArray(concatHorizontally(this.storage, a.storage))
  }

  /**
   *
   */
  override fun concatV(a: NDArrayInterface): JBlasArray { (a as JBlasArray)
    return JBlasArray(concatVertically(this.storage, a.storage))
  }

  /**
   * Return a one-dimensional JBlasArray sub-vector of a vertical vector
   */
  override fun getRange(a: Int, b: Int): JBlasArray {
    require(this.shape.dim2 == 1)
    return JBlasArray(this.storage.getRange(a, b))
  }

  /**
   *
   */
  override fun zerosLike(): JBlasArray {
    return JBlasArray(DoubleMatrix.zeros(this.shape.dim1, shape.dim2))
  }

  /**
   * @param a a NDArrayInterface
   * @param tolerance a must be in the range [a - tolerance, a + tolerance] to return True
   *
   * @return a Boolean which indicates if a is equal to be within the tolerance
   */
  override fun equals(a: NDArrayInterface, tolerance: Double): Boolean {
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
    return other is JBlasArray && this.equals(other)
  }

  /**
   *
   */
  override fun hashCode(): Int {
    return this.storage.hashCode()
  }
}
