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
import com.kotlinnlp.simplednn.simplemath.ndarray.*
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray
import org.jblas.MatrixFunctions.abs

/**
 * [NDArray] with dense values (implemented using JBlas)
 */
class DenseNDArray(private val storage: DoubleMatrix) : NDArray<DenseNDArray> {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
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
  override val t: DenseNDArray
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
    return DenseNDArrayFactory.arrayOf(listOf(values.toArray()))
  }

  /**
   * Get the i-th column
   *
   * @param i the index of the column to be returned
   *
   * @return the selected column as a new DenseNDArray
   */
  override fun getColumn(i: Int): DenseNDArray = DenseNDArray(this.storage.getColumn(i))

  /**
   * Get a one-dimensional DenseNDArray sub-vector of a vertical vector.
   *
   * @param a the start index of the range (inclusive)
   * @param b the end index of the range (exclusive)
   *
   * @return the sub-array
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
   * Fill the array with ones.
   */
  fun ones(): DenseNDArray {
    this.storage.fill(1.0)
    return this
  }

  /**
   * @return a new [DenseNDArray] with the same shape of this, filled with zeros.
   */
  override fun zerosLike(): DenseNDArray = DenseNDArray(DoubleMatrix.zeros(this.shape.dim1, shape.dim2))

  /**
   * @return a new [DenseNDArray] with the same shape of this, filled with ones.
   */
  fun onesLike(): DenseNDArray = DenseNDArray(DoubleMatrix.ones(this.shape.dim1, shape.dim2))

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
   * Assign the values of [a] to this [DenseNDArray] (it works also among rows and columns vectors).
   *
   * @param a a generic [NDArray]
   *
   * @return this [DenseNDArray]
   */
  override fun assignValues(a: NDArray<*>): DenseNDArray {
    require(this.shape == a.shape ||
      (this.isVector && a.isVector && this.length == a.length))

    when(a) {
      is DenseNDArray -> this.assignValues(a)
      is SparseNDArray -> this.assignValues(a)
      is SparseBinaryNDArray -> TODO("not implemented")
    }

    return this
  }

  /**
   * Assign the values of [a] to this [DenseNDArray] (it works also among rows and columns vectors).
   *
   * @param a a [DenseNDArray]
   */
  private fun assignValues(a: DenseNDArray) {
    System.arraycopy(a.storage.data, 0, this.storage.data, 0, this.length)
  }

  /**
   * Assign the values of [a] to this [DenseNDArray] (it works also among rows and columns vectors).
   *
   * @param a a [SparseNDArray]
   */
  private fun assignValues(a: SparseNDArray) {

    this.zeros()

    for (k in 0 until a.values.size) {
      this[a.rowIndices[k], a.colIndices[k]] = a.values[k]
    }
  }

  /**
   *
   */
  override fun assignValues(a: NDArray<*>, mask: NDArrayMask): DenseNDArray {
    require(a.shape == this.shape) { "Arrays with different size" }

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
  private fun assignValues(a: DenseNDArray, mask: NDArrayMask): DenseNDArray {

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
  private fun assignValues(a: SparseNDArray, mask: NDArrayMask): DenseNDArray {
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
  override fun sum(n: Double): DenseNDArray = DenseNDArray(this.storage.add(n))

  /**
   *
   */
  override fun sum(a: DenseNDArray): DenseNDArray = DenseNDArray(this.storage.add(a.storage))

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
  private fun assignSum(a: SparseNDArray): DenseNDArray {

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
  override fun sub(n: Double): DenseNDArray = DenseNDArray(this.storage.sub(n))

  /**
   *
   */
  override fun sub(a: DenseNDArray): DenseNDArray = DenseNDArray(this.storage.sub(a.storage))

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
  private fun assignSub(a: SparseNDArray): DenseNDArray {
    require(a.shape == this.shape) { "Arrays with different size" }

    for (k in 0 until a.values.size) {
      this[a.rowIndices[k], a.colIndices[k]] -= a.values[k]
    }

    return this
  }

  /**
   *
   */
  override fun reverseSub(n: Double): DenseNDArray = DenseNDArray(this.storage.rsub(n))

  /**
   * Dot product between this [DenseNDArray] and a [DenseNDArray] masked by [mask].
   *
   * @param a the [DenseNDArray] by which is calculated the dot product
   * @param mask the mask applied to a
   *
   * @return a new [DenseNDArray]
   */
  fun dot(a: DenseNDArray, mask: NDArrayMask): DenseNDArray {
    require(this.columns == a.rows)
    require(this.rows == 1) // TODO: extend to all shapes

    val ret = DenseNDArrayFactory.zeros(shape = Shape(this.rows, a.columns))

    (0 until a.columns).forEach { j ->
      ret[j] = mask.dim1.sumByDouble { k -> this[k] * a[k, j] }
    }

    return ret
  }

  /**
   *
   */
  override fun dot(a: NDArray<*>): DenseNDArray = when(a) {
    is DenseNDArray -> this.dot(a)
    is SparseNDArray -> TODO("not implemented")
    is SparseBinaryNDArray -> this.dot(a)
    else -> throw RuntimeException("Invalid NDArray type")
  }

  /**
   *
   */
  private fun dot(a: DenseNDArray): DenseNDArray = DenseNDArray(this.storage.mmul(a.storage))

  /**
   *
   */
  private fun dot(a: SparseBinaryNDArray): DenseNDArray {
    require(this.columns == a.rows)

    val res = DenseNDArrayFactory.zeros(shape = Shape(this.rows, a.columns))

    when {
      a.rows == 1 -> // Column vector (dot) row vector
        for (j in a.activeIndicesByColumn.keys) {
          for (i in 0 until this.rows) {
            res.storage.put(i, j, this[i])
          }
        }
      a.columns == 1 -> // n-dim array (dot) column vector
        for (i in 0 until this.rows) {
          res.storage.put(i, a.activeIndicesByRow.keys.sumByDouble { this[i, it] })
        }
      else -> // n-dim array (dot) n-dim array
        TODO("not implemented")
    }

    return res
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
  private fun assignDot(a: DenseNDArray, b: SparseBinaryNDArray): DenseNDArray {
    require(a.rows == this.rows && b.columns == this.columns && a.columns == b.rows)

    this.zeros()

    when {
      b.rows == 1 -> // Column vector (dot) row vector
        for (j in b.activeIndicesByColumn.keys) {
          for (i in 0 until a.rows) {
            this.storage.put(i, j, a[i])
          }
        }
      b.columns == 1 -> // n-dim array (dot) column vector
        for (i in 0 until a.rows) {
          this.storage.put(i, b.activeIndicesByRow.keys.sumByDouble { a[i, it] })
        }
      else -> // n-dim array (dot) n-dim array
        TODO("not implemented")
    }

    return this
  }

  /**
   *
   */
  fun assignDot(a: DenseNDArray, b: DenseNDArray, aMask: NDArrayMask): DenseNDArray {
    require(a.rows == this.rows && b.columns == this.columns && a.columns == b.rows)

    this.zeros()

    for (bCol in 0 until b.shape.dim2) {
      for ((aRow, aCol) in aMask) {
        this[aRow, bCol] += a[aRow, aCol] * b[aCol, bCol]
      }
    }

    return this
  }

  /**
   *
   */
  override fun prod(n: Double): DenseNDArray = DenseNDArray(this.storage.mul(n))

  /**
   *
   */
  override fun prod(a: NDArray<*>): DenseNDArray = when(a) {
    is DenseNDArray -> this.prod(a)
    is SparseNDArray -> TODO("not implemented")
    is SparseBinaryNDArray -> TODO("not implemented")
    else -> throw RuntimeException("Invalid NDArray type")
  }

  /**
   * Product by a [DenseNDArray] with the same shape or a compatible column vector (each column is multiplied
   * by the given vector). It works also against a row and a column vector.
   *
   * @param a the [DenseNDArray] by which this [DenseNDArray] will be multiplied
   *
   * @return a new [DenseNDArray] containing the product between this [DenseNDArray] and [a]
   */
  private fun prod(a: DenseNDArray): DenseNDArray {
    require(a.shape == this.shape ||
      (a.columns == 1 && a.rows == this.rows) ||
      (a.isVector && this.isVector && a.length == this.length)) { "Arrays with not compatible size" }

    return if (a.shape == this.shape)
      DenseNDArray(this.storage.mul(a.storage))

    else
      DenseNDArray(DoubleMatrix(
        this.storage.rows,
        this.storage.columns,
        *DoubleArray(
          size = this.length,
          init = { k -> this.storage[k] * a[k % a.length] } // linear indexing
        )
      ))
  }

  /**
   *
   */
  override fun prod(n: Double, mask: NDArrayMask): SparseNDArray {

    val values = DoubleArray(size = mask.size, init = { this.storage[mask.dim1[it], mask.dim2[it]] * n })

    return SparseNDArray(shape = this.shape, values = values, rows = mask.dim1, columns = mask.dim2)
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
  fun assignProd(a: NDArray<*>): DenseNDArray = when(a) {
    is DenseNDArray -> this.assignProd(a)
    is SparseNDArray -> this.assignProd(a)
    is SparseBinaryNDArray -> this.assignProd(a)
    else -> throw RuntimeException("Invalid NDArray type")
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
  private fun assignProd(a: SparseNDArray): DenseNDArray {

    val newValues: List<Pair<Indices, Double>> = a.map { (indices, value) ->
      Pair(indices, value * this[indices.first, indices.second])
    }

    this.zeros()

    newValues.forEach { (indices, newValue) -> this[indices.first, indices.second] = newValue }

    return this
  }

  /**
   *
   */
  private fun assignProd(a: SparseBinaryNDArray): DenseNDArray {

    val values: List<Pair<Indices, Double>> = a.map { indices -> Pair(indices, this[indices.first, indices.second]) }

    this.zeros()

    values.forEach { (indices, values) -> this[indices.first, indices.second] = values }

    return this
  }

  /**
   *
   */
  override fun div(n: Double): DenseNDArray = DenseNDArray(this.storage.div(n))

  /**
   *
   */
  override fun div(a: NDArray<*>): DenseNDArray = when(a) {
    is DenseNDArray -> DenseNDArray(this.storage.div(a.storage))
    is SparseNDArray -> TODO("not implemented")
    is SparseBinaryNDArray -> TODO("not implemented")
    else -> throw RuntimeException("Invalid NDArray type")
  }

  /**
   *
   */
  override fun div(a: SparseNDArray): SparseNDArray {
    require(a.shape == this.shape) { "Arrays with different size" }

    return SparseNDArray(
      shape = this.shape.copy(),
      values = DoubleArray(size = a.values.size, init = { i -> this[a.rowIndices[i], a.colIndices[i]] / a.values[i]}),
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
   *
   */
  override fun abs() = DenseNDArray(storage = abs(this.storage))

  /**
   * Sign function.
   *
   * @return a new [DenseNDArray] containing the results of the function sign() applied element-wise
   */
  override fun sign(): DenseNDArray = DenseNDArray(MatrixFunctions.signum(this.storage))

  /**
   * Non-zero sign function.
   *
   * @return a new [DenseNDArray] containing +1 or -1 values depending on the sign element-wise (+1 if the value is 0)
   */
  fun nonZeroSign(): DenseNDArray
    = DenseNDArray(MatrixFunctions.signum(MatrixFunctions.signum(this.storage).addi(0.1)))

  /**
   *
   */
  override fun sqrt(): DenseNDArray = DenseNDArray(MatrixFunctions.sqrt(this.storage))

  /**
   * Square root of this [DenseNDArray] masked by [mask]
   *
   * @param mask the mask to apply
   *
   * @return a [SparseNDArray]
   */
  override fun sqrt(mask: NDArrayMask): SparseNDArray {

    val values = DoubleArray(size = mask.size, init = { Math.sqrt(this.storage[mask.dim1[it], mask.dim2[it]]) })

    return SparseNDArray(shape = this.shape, values = values, rows = mask.dim1, columns = mask.dim2)
  }

  /**
   * Power.
   *
   * @param power the exponent
   *
   * @return a new [DenseNDArray] containing the values of this to the power of [power]
   */
  override fun pow(power: Double): DenseNDArray = DenseNDArray(MatrixFunctions.pow(this.storage, power))

  /**
   * In-place power.
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
   * Logarithm with base 10.
   *
   * @return a new [DenseNDArray] containing the element-wise logarithm with base 10 of this array
   */
  override fun log10(): DenseNDArray {

    require((0 until this.length).all { i -> this[i] != 0.0 })

    return DenseNDArray(MatrixFunctions.log10(this.storage))
  }

  /**
   * In-place logarithm with base 10.
   *
   * @return this [DenseNDArray] after having applied the logarithm with base 10 to its values
   */
  override fun assignLog10(): DenseNDArray {

    require((0 until this.length).all { i -> this[i] != 0.0 })

    MatrixFunctions.log10i(this.storage)

    return this
  }

  /**
   * Natural logarithm.
   *
   * @return a new [DenseNDArray] containing the element-wise natural logarithm of this array
   */
  override fun ln(): DenseNDArray {

    require((0 until this.length).all { i -> this[i] != 0.0 })

    return DenseNDArray(MatrixFunctions.log(this.storage))
  }

  /**
   * In-place logarithm with base 10.
   *
   * @return this [DenseNDArray] after having applied the logarithm with base 10 to its values
   */
  override fun assignLn(): DenseNDArray {

    require((0 until this.length).all { i -> this[i] != 0.0 })

    MatrixFunctions.logi(this.storage)

    return this
  }

  /**
   * The norm (L1 distance) of this NDArray.
   *
   * @return the norm
   */
  override fun norm(): Double = (0 until this.length).sumByDouble { i -> abs(this[i]) }

  /**
   * The Euclidean norm of this DenseNDArray.
   *
   * @return the euclidean norm
   */
  override fun norm2(): Double =
    this.storage.distance2(DoubleMatrix.zeros(this.shape.dim1, shape.dim2))

  /**
   * @return the maximum value of this NDArray
   **/
  fun max(): Double = this.storage.max()

  /**
   * Get the index of the highest value eventually skipping the element at the given [exceptIndex] when it is >= 0.
   *
   * @param exceptIndex the index to exclude
   *
   * @return the index of the maximum value (-1 if empty)
   **/
  override fun argMaxIndex(exceptIndex: Int): Int {

    var maxIndex: Int = -1
    var maxValue: Double? = null

    (0 until this.length).forEach { i ->

      if (i != exceptIndex) {

        val value = this[i]

        if (maxValue == null || value > maxValue!!) {
          maxValue = value
          maxIndex = i
        }
      }
    }

    return maxIndex
  }

  /**
   * Get the index of the highest value skipping all the elements at the indices in given set.
   *
   * @param exceptIndices the set of indices to exclude
   *
   * @return the index of the maximum value (-1 if empty)
   **/
  override fun argMaxIndex(exceptIndices: Set<Int>): Int {

    var maxIndex: Int = -1
    var maxValue: Double? = null

    (0 until this.length).forEach { i ->

      if (i !in exceptIndices) {

        val value = this[i]

        if (maxValue == null || value > maxValue!!) {
          maxValue = value
          maxIndex = i
        }
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
  override fun concatH(a: DenseNDArray): DenseNDArray = DenseNDArray(concatHorizontally(this.storage, a.storage))

  /**
   *
   */
  override fun concatV(a: DenseNDArray): DenseNDArray = DenseNDArray(concatVertically(this.storage, a.storage))

  /**
   * Split this NDArray into more NDArrays.
   *
   * If the number of arguments is one, split this NDArray into multiple NDArray each with length [splittingLength].
   * If there are multiple arguments, split this NDArray according to the length of each [splittingLength] element.
   *
   * @param splittingLength the length(s) for sub-array division
   *
   * @return a list containing the split values
   */
  override fun splitV(vararg splittingLength: Int): List<DenseNDArray> =
    if (splittingLength.size == 1)
      this.splitVSingleSegment(splittingLength.first())
    else
      this.splitVMultipleSegments(splittingLength)

  /**
   * Split this NDArray into multiple NDArray each with length [splittingLength]
   *
   * @param splittingLength the length for sub-array division
   *
   * @return a list containing the split values
   */
  private fun splitVSingleSegment(splittingLength: Int): List<DenseNDArray> {

    require(this.length % splittingLength == 0) {
      "The length of the array must be a multiple of the splitting length"
    }

    return List(
      size = this.length / splittingLength,
      init = {
        val startIndex = it * splittingLength
        this.getRange(startIndex, startIndex + splittingLength)
      }
    )
  }

  /**
   * Split this NDArray according to the length of each [splittingLength] element.
   *
   * @param splittingLength the lengths for sub-array division
   *
   * @return a list containing the split values
   */
  private fun splitVMultipleSegments(splittingLength: IntArray): List<DenseNDArray> {

    require(splittingLength.sum() == this.length) {
      "The length of the array must be equal to the sum of each splitting length"
    }

    var offset = 0

    return List(
      size = splittingLength.size,
      init = {
        val startIndex = offset
        offset = startIndex + splittingLength[it]
        this.getRange(startIndex, offset)
      }
    )
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
  override fun equals(other: Any?): Boolean = other is DenseNDArray && this.equals(other)

  /**
   *
   */
  override fun hashCode(): Int = this.storage.hashCode()

  /**
   *
   */
  fun maskBy(mask: NDArrayMask): SparseNDArray = SparseNDArray(
    shape = this.shape,
    values = DoubleArray(size = mask.size, init = { i -> this.storage[mask.dim1[i], mask.dim2[i]] }),
    rows = mask.dim1,
    columns = mask.dim2
  )

  /**
   *
   */
  fun toDoubleArray(): DoubleArray = this.storage.dup().data

  /**
   * @param reverse whether to sort in descending order
   *
   * @return a permutation of indices which makes the 1-D array sorted.
   */
  fun argSort(reverse: Boolean = false): IntArray {

    require(this.isVector) { "Operation supported only by vectors." }

    val doubleArray = this.storage.data
    val comparator = Comparator(IndexedDoubleValue::compareTo)
    val indexedValues = Array(doubleArray.size) { IndexedDoubleValue(it, this[it]) }

    indexedValues.sortWith(if (reverse) comparator.reversed() else comparator)
    return IntArray(doubleArray.size) { indexedValues[it].index }
  }

  /**
   * A version of [IndexedValue] specialized to [Double].
   */
  private data class IndexedDoubleValue(val index: Int, val value: Double) : Comparable<IndexedDoubleValue> {

    override fun compareTo(other: IndexedDoubleValue): Int = this.value.compareTo(other.value).let { res ->
      return if (res == 0) index.compareTo(other.index) else res
    }
  }
}
