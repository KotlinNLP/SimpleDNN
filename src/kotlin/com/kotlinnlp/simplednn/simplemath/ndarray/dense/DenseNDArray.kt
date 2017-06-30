/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.simplemath.ndarray.dense

import org.jblas.DoubleMatrix
import org.jblas.MatrixFunctions
import org.jblas.DoubleMatrix.concatHorizontally
import org.jblas.DoubleMatrix.concatVertically
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.simplemath.equals
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArrayMask
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray

/**
 * [NDArray] with dense values (implemented using JBlas)
 */
class DenseNDArray(private val storage: DoubleMatrix) : NDArray<DenseNDArray> {

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
  override val factory = DenseNDArrayFactory

  /**
   * Whether the array is a row or a column vector
   */
  override val isVector: Boolean
    get() = this.storage.rows == 1 || this.storage.columns == 1

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
  override val rows: Int
    get() = this.storage.rows

  /**
   *
   */
  override val columns: Int
    get() = this.storage.columns

  /**
   *
   */
  override val length: Int
    get() = this.storage.length

  /**
   *
   * @return
   */
  override val shape: Shape
    get() = Shape(this.rows, this.columns)

  /**
   *
   */
  override val T: DenseNDArray
    get() = DenseNDArray(this.storage.transpose())

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
   * @return the selected row as a new DenseNDArray
   */
  override fun getRow(i: Int): DenseNDArray {
    val values = this.storage.getRow(i)
    return DenseNDArrayFactory.arrayOf(arrayOf<DoubleArray>(values.toArray()))
  }

  /**
   * Get the i-th column
   *
   * @param i the index of the column to be returned
   *
   * @return the selected column as a new DenseNDArray
   */
  override fun getColumn(i: Int): DenseNDArray {
    return DenseNDArray(this.storage.getColumn(i))
  }

  /**
   * Return a one-dimensional DenseNDArray sub-vector of a vertical vector
   */
  override fun getRange(a: Int, b: Int): DenseNDArray {
    require(this.shape.dim2 == 1)
    return DenseNDArray(this.storage.getRange(a, b))
  }

  /**
   *
   */
  override fun zeros(): DenseNDArray {
    this.storage.fill(0.0)
    return this
  }

  /**
   *
   */
  override fun zerosLike(): DenseNDArray {
    return DenseNDArray(DoubleMatrix.zeros(this.shape.dim1, shape.dim2))
  }

  /**
   *
   */
  override fun copy(): DenseNDArray = DenseNDArray(this.storage.dup())

  /**
   *
   */
  override fun assignValues(n: Double): DenseNDArray {
    this.storage.fill(n)
    return this
  }

  /**
   * Assign the values of a to this DenseNDArray (it works also among rows and columns vectors)
   */
  override fun assignValues(a: NDArray<*>): DenseNDArray {
    require(this.shape == a.shape ||
      (this.isVector && a.isVector && this.length == a.length))

    when(a) {
      is DenseNDArray -> System.arraycopy(a.storage.data, 0, this.storage.data, 0, this.length)
      is SparseNDArray -> {
        this.zeros()
        for (k in 0 until a.values.size) {
          this[a.rowIndices[k], a.colIndices[k]] = a.values[k]
        }
      }
      is SparseBinaryNDArray -> TODO("not implemented")
    }

    return this
  }

  /**
   *
   */
  override fun assignValues(a: NDArray<*>, mask: NDArrayMask): DenseNDArray {

    when(a) {
      is DenseNDArray -> this.assignValues(a, mask)
      is SparseNDArray -> this.assignValues(a, mask)
      is SparseBinaryNDArray -> TODO("not implemented")
    }

    return this
  }

  /**
   *
   */
  fun assignValues(a: DenseNDArray, mask: NDArrayMask): DenseNDArray {
    require(a.shape == this.shape) { "Arrays with different size" }
    require(mask.shape == this.shape) { "Mask has not compatible shape" }

    for (index in 0 until mask.size) {
      val i = mask.dim1[index]
      val j = mask.dim2[index]
      this.storage.put(i, j, a[i, j])
    }

    return this
  }

  /**
   *
   */
  fun assignValues(a: SparseNDArray, mask: NDArrayMask): DenseNDArray {
    require(a.shape == this.shape) { "Arrays with different size" }
    require(mask.shape == this.shape) { "Mask has not compatible shape" }
    require(a.values.size == mask.size) { "Mask has a different number of active values respect of a" }

    for (index in 0 until mask.size) {
      val i = mask.dim1[index]
      val j = mask.dim2[index]
      this.storage.put(i, j, a.values[index])
    }

    return this
  }

  /**
   *
   */
  override fun sum(): Double = this.storage.sum()

  /**
   *
   */
  override fun sum(n: Double): DenseNDArray {
    return DenseNDArray(this.storage.add(n))
  }

  /**
   *
   */
  override fun sum(a: DenseNDArray): DenseNDArray {
    return DenseNDArray(this.storage.add(a.storage))
  }

  /**
   *
   */
  override fun assignSum(n: Double): DenseNDArray {
    this.storage.addi(n)
    return this
  }

  /**
   * Assign a to this DenseNDArray (it works also among rows and columns vectors)
   */
  override fun assignSum(a: NDArray<*>): DenseNDArray {

    when(a) {
      is DenseNDArray -> this.storage.addi(a.storage)
      is SparseNDArray -> this.assignSum(a)
      is SparseBinaryNDArray -> TODO("not implemented")
    }

    return this
  }

  /**
   *
   */
  fun assignSum(a: SparseNDArray): DenseNDArray {

    for (index in 0 until a.values.size) {
      this.storage.put(
        a.rowIndices[index],
        a.colIndices[index],
        this.storage[a.rowIndices[index], a.colIndices[index]] + a.values[index]
      )
    }

    return this
  }

  /**
   *
   */
  override fun assignSum(a: DenseNDArray, n: Double): DenseNDArray {
    a.storage.addi(n, this.storage)
    return this
  }

  /**
   * Assign a + b to this DenseNDArray (it works also among rows and columns vectors)
   */
  override fun assignSum(a: DenseNDArray, b: DenseNDArray): DenseNDArray {
    a.storage.addi(b.storage, this.storage)
    return this
  }

  /**
   *
   */
  override fun sub(n: Double): DenseNDArray {
    return DenseNDArray(this.storage.sub(n))
  }

  /**
   *
   */
  override fun sub(a: DenseNDArray): DenseNDArray {
    return DenseNDArray(this.storage.sub(a.storage))
  }

  /**
   * In-place subtraction by number
   */
  override fun assignSub(n: Double): DenseNDArray {
    this.storage.subi(n)
    return this
  }

  /**
   *
   */
  override fun assignSub(a: NDArray<*>): DenseNDArray {

    when(a) {
      is DenseNDArray -> this.storage.subi(a.storage)
      is SparseNDArray -> this.assignSub(a)
      is SparseBinaryNDArray -> TODO("not implemented")
    }

    return this
  }

  /**
   *
   */
  fun assignSub(a: SparseNDArray): DenseNDArray {
    require(a.shape == this.shape) { "Arrays with different size" }

    for (k in 0 until a.values.size) {
      this[a.rowIndices[k], a.colIndices[k]] -= a.values[k]
    }

    return this
  }

  /**
   *
   */
  override fun reverseSub(n: Double): DenseNDArray {
    return DenseNDArray(this.storage.rsub(n))
  }

  /**
   *
   */
  override fun dot(a: DenseNDArray): DenseNDArray {
    return DenseNDArray(this.storage.mmul(a.storage))
  }

  /**
   * Dot product between this [DenseNDArray] and a [DenseNDArray] masked by [mask]
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
  override fun assignDot(a: DenseNDArray, b: DenseNDArray): DenseNDArray {
    require(a.rows == this.rows && b.columns == this.columns)
    a.storage.mmuli(b.storage, this.storage)
    return this
  }

  /**
   *
   */
  override fun assignDot(a: DenseNDArray, b: NDArray<*>): DenseNDArray {

    when(b) {
      is DenseNDArray -> this.assignDot(a, b)
      is SparseNDArray -> TODO("not implemented")
      is SparseBinaryNDArray -> this.assignDot(a, b)
    }

    return this
  }

  /**
   *
   */
  fun assignDot(a: DenseNDArray, b: SparseBinaryNDArray): DenseNDArray {
    require(a.rows == this.rows && b.columns == this.columns && a.columns == b.rows)

    this.zeros()

    if (b.rows == 1) {
      // Column vector (dot) row vector
      for (j in b.activeIndicesByColumn.keys) {
        for (i in 0 until a.rows) {
          this.storage.put(i, j, a[i])
        }
      }

    } else if (b.columns == 1) {
      // n-dim array (dot) column vector
      for (i in 0 until a.rows) {
        this.storage.put(i, b.activeIndicesByRow.keys.sumByDouble { a[i, it] })
      }

    } else {
      // n-dim array (dot) n-dim array
      TODO("not implemented")
    }

    return this
  }

  /**
   *
   */
  override fun prod(n: Double): DenseNDArray {
    return DenseNDArray(this.storage.mul(n))
  }

  /**
   *
   */
  override fun prod(a: NDArray<*>): DenseNDArray {

    return when(a) {
      is DenseNDArray -> DenseNDArray(this.storage.mul(a.storage))
      is SparseNDArray -> TODO("not implemented")
      is SparseBinaryNDArray -> TODO("not implemented")
      else -> throw RuntimeException("Invalid NDArray type")
    }
  }

  /**
   *
   */
  override fun prod(n: Double, mask: NDArrayMask): SparseNDArray {

    val values = Array(size = mask.size, init = { this.storage[mask.dim1[it], mask.dim2[it]] * n })

    return SparseNDArray(shape = mask.shape, values = values, rows = mask.dim1, columns = mask.dim2)
  }

  /**
   *
   */
  override fun assignProd(n: Double): DenseNDArray {
    this.storage.muli(n)
    return this
  }

  /**
   *
   */
  override fun assignProd(n: Double, mask: NDArrayMask): DenseNDArray {

    for (index in 0 until mask.size) {
      this.storage.put(mask.dim1[index], mask.dim2[index], this.storage[mask.dim1[index], mask.dim2[index]] * n)
    }

    return this
  }

  /**
   *
   */
  override fun assignProd(a: DenseNDArray, n: Double): DenseNDArray {
    a.storage.muli(n, this.storage)
    return this
  }

  /**
   *
   */
  override fun assignProd(a: DenseNDArray, b: DenseNDArray): DenseNDArray {
    a.storage.muli(b.storage, this.storage)
    return this
  }

  /**
   *
   */
  override fun assignProd(a: DenseNDArray): DenseNDArray {
    this.storage.muli(a.storage)
    return this
  }

  /**
   *
   */
  override fun div(n: Double): DenseNDArray {
    return DenseNDArray(this.storage.div(n))
  }

  /**
   *
   */
  override fun div(a: NDArray<*>): DenseNDArray {

    return when(a) {
      is DenseNDArray -> DenseNDArray(this.storage.div(a.storage))
      is SparseNDArray -> TODO("not implemented")
      is SparseBinaryNDArray -> TODO("not implemented")
      else -> throw RuntimeException("Invalid NDArray type")
    }
  }

  /**
   *
   */
  override fun div(a: SparseNDArray): SparseNDArray {
    require(a.shape == this.shape) { "Arrays with different size" }

    return SparseNDArray(
      shape = this.shape.copy(),
      values = Array(size = a.values.size, init = { i -> this[a.rowIndices[i], a.colIndices[i]] / a.values[i]}),
      rows = a.rowIndices.copyOf(),
      columns = a.colIndices.copyOf()
    )
  }

  /**
   *
   */
  override fun assignDiv(n: Double): DenseNDArray {
    this.storage.divi(n)
    return this
  }

  /**
   *
   */
  override fun assignDiv(a: DenseNDArray): DenseNDArray {
    this.storage.divi(a.storage)
    return this
  }

  /**
   *
   */
  override fun avg(): Double = this.storage.mean()

  /**
   * Sign function.
   *
   * @return a new [DenseNDArray] containing the results of the function sign() applied element-wise
   */
  override fun sign(): DenseNDArray {
    return DenseNDArray(MatrixFunctions.signum(this.storage))
  }

  /**
   * Non-zero sign function.
   *
   * @return a new [DenseNDArray] containing +1 or -1 values depending on the sign element-wise (+1 if the value is 0)
   */
  fun nonZeroSign(): DenseNDArray {
    return DenseNDArray(MatrixFunctions.signum(MatrixFunctions.signum(this.storage).addi(0.1)))
  }

  /**
   *
   */
  override fun sqrt(): DenseNDArray {
    return DenseNDArray(MatrixFunctions.sqrt(this.storage))
  }

  /**
   * Square root of this [DenseNDArray] masked by [mask]
   *
   * @param mask the mask to apply
   *
   * @return a [SparseNDArray]
   */
  override fun sqrt(mask: NDArrayMask): SparseNDArray {

    val values = Array(size = mask.size, init = { Math.sqrt(this.storage[mask.dim1[it], mask.dim2[it]]) })

    return SparseNDArray(shape = mask.shape, values = values, rows = mask.dim1, columns = mask.dim2)
  }


  /**
   * Power
   *
   * @param power the exponent
   *
   * @return a new [DenseNDArray] containing the values of this to the power of [power]
   */
  override fun pow(power: Double): DenseNDArray {
    return DenseNDArray(MatrixFunctions.pow(this.storage, power))
  }

  /**
   * In-place power
   *
   * @param power the exponent
   *
   * @return this [DenseNDArray] to the power of [power]
   */
  override fun assignPow(power: Double): DenseNDArray {
    MatrixFunctions.powi(this.storage, power)
    return this
  }

  /**
   * Euclidean norm of this DenseNDArray
   *
   * @return the euclidean norm
   */
  override fun norm2(): Double {
    val zeros = this.zerosLike()
    return this.storage.distance2(zeros.storage)
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
   * Round values to Int
   *
   * @param threshold a value is rounded to the next Int if is >= [threshold], to the previous otherwise
   *
   * @return a new DenseNDArray with the values of the current one rounded to Int
   */
  override fun roundInt(threshold: Double): DenseNDArray {

    val out = DenseNDArrayFactory.emptyArray(this.shape)
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
  override fun assignRoundInt(threshold: Double): DenseNDArray {

    val floorValues = MatrixFunctions.floor(this.storage)

    for (i in 0 until this.length) {
      this[i] = if (this.storage[i] < threshold) floorValues[i] else floorValues[i] + 1
    }

    return this
  }

  /**
   *
   */
  override fun randomize(randomGenerator: RandomGenerator): DenseNDArray {
    for (i in 0 until this.length) this[i] = randomGenerator.next() // i: linear index
    return this
  }

  /**
   *
   */
  override fun concatH(a: DenseNDArray): DenseNDArray {
    return DenseNDArray(concatHorizontally(this.storage, a.storage))
  }

  /**
   *
   */
  override fun concatV(a: DenseNDArray): DenseNDArray {
    return DenseNDArray(concatVertically(this.storage, a.storage))
  }

  /**
   * Splits this NDArray into multiple NDArray each with length [splittingLength]
   *
   * @param splittingLength the length for sub-array division
   *
   * @return an Array containing the split values
   */
  override fun splitV(splittingLength: Int): Array<DenseNDArray>{

    require(this.length % splittingLength == 0){
      "The length of the array must be a multiple of the splitting length"
    }

    val result = arrayOfNulls<DenseNDArray>(this.length / splittingLength)

    var start = 0
    var end = splittingLength
    var i = 0

    while (start < this.length){
      result[i] = this.getRange(start, end)
      start = end
      end += splittingLength
      i++
    }

    return result.requireNoNulls()
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

  /**
   *
   */
  fun maskBy(mask: NDArrayMask): SparseNDArray = SparseNDArray(
    shape = this.shape,
    values = Array(size = mask.size, init = { i -> this.storage[mask.dim1[i], mask.dim2[i]] }),
    rows = mask.dim1,
    columns = mask.dim2
  )

  /**
   *
   */
  fun toDoubleArray(): DoubleArray {
    return this.storage.dup().data
  }
}
