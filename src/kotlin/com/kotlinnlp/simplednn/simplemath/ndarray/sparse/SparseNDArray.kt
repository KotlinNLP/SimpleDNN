/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.simplemath.ndarray.sparse

import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.simplemath.equals
import com.kotlinnlp.simplednn.simplemath.ndarray.*
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import java.util.*

/**
 *
 */
class SparseNDArray(override val shape: Shape) : NDArray<SparseNDArray>, Iterable<SparseEntry>  {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Secondary Factory.
     */
    operator fun invoke(shape: Shape, values: Array<Double>, rows: Array<Int>, columns: Array<Int>): SparseNDArray {

      val array = SparseNDArray(shape = shape)

      array.values = values
      array.rowIndices = rows
      array.colIndices = columns

      return array
    }
  }

  /**
   *
   */
  inner class LinearIterator : Iterator<SparseEntry>  {

    /**
     *
     */
    private var curIndex: Int = 0

    /**
     *
     */
    override fun hasNext(): Boolean = this.curIndex < this@SparseNDArray.values.size

    /**
     *
     */
    override fun next(): SparseEntry {

      val value = this@SparseNDArray.values[this.curIndex]
      val indices = Pair(
        this@SparseNDArray.rowIndices[this.curIndex],
        this@SparseNDArray.colIndices[this.curIndex]
      )

      this.curIndex++

      return Pair(indices, value)
    }
  }

  /**
   * Iterator over active indices with the related values
   */
  override fun iterator(): Iterator<SparseEntry> {
    return LinearIterator()
  }

  /**
   *
   */
  var values = arrayOf<Double>()
    private set

  /**
   *
   */
  var rowIndices = arrayOf<Int>()
    private set

  /**
   *
   */
  var colIndices = arrayOf<Int>()
    private set

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
  override val isOneHotEncoder: Boolean
    get() = TODO("not implemented")

  /**
   *
   */
  override val rows: Int = this.shape.dim1

  /**
   *
   */
  override val columns: Int = this.shape.dim2

  /**
   *
   */
  override val length: Int = this.rows * this.columns

  /**
   * Transpose
   */
  override val T: SparseNDArray get() = SparseNDArray(
    shape = this.shape.inverse,
    values = this.values.copyOf(),
    rows = this.colIndices.copyOf(),
    columns = this.rowIndices.copyOf()
  )

  /**
   *
   */
  val mask: NDArrayMask get() = NDArrayMask(dim1 = this.rowIndices, dim2 = this.colIndices, shape = this.shape)

  /**
   *
   */
  override fun get(i: Int): Double {
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
  override fun set(i: Int, value: Number) {
    require(i < this.length)

    if (this.rows == 1) {
      this[0, i] = value

    } else if (this.columns == 1) {
      this[i, 0] = value

    } else {
      this[i / this.columns, i % this.columns] = value
    }
  }

  /**
   *
   */
  override fun set(i: Int, j: Int, value: Number) {
    require(i < this.rows && j < this.columns)

    if (value != 0.0) {
      this.setElement(row = i, col = j, value = value.toDouble())

    } else {
      TODO("not implemented")
    }
  }

  /**
   *
   */
  private fun setElement(row: Int, col: Int, value: Double) {

    var index: Int = 0

    while (index < this.values.size && this.colIndices[index] != col) index++
    while (index < this.values.size && this.rowIndices[index] != row) index++

    if (index > this.values.size) {
      throw RuntimeException("Cannot set value at indices not already active")

    } else {
      this.values[index] = value
    }
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
   * Return a one-dimensional NDArray sub-vector of a vertical vector
   */
  override fun getRange(a: Int, b: Int): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun zeros(): SparseNDArray {
    this.values = arrayOf()
    this.rowIndices = arrayOf()
    this.colIndices = arrayOf()
    return this
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
  override fun copy(): SparseNDArray = SparseNDArray(
    shape = this.shape.copy(),
    values = this.values.copyOf(),
    rows = this.rowIndices.copyOf(),
    columns = this.colIndices.copyOf()
  )

  /**
   *
   */
  override fun assignValues(n: Double): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  fun assignValues(a: SparseNDArray): SparseNDArray {
    require(a.shape == this.shape)

    this.values = a.values.copyOf()
    this.rowIndices = a.rowIndices.copyOf()
    this.colIndices = a.colIndices.copyOf()

    return this
  }

  /**
   *
   */
  fun assignValues(values: Array<Double>, rowIndices: Array<Int>, colIndices: Array<Int>): SparseNDArray {
    require(rowIndices.all{ i -> rowIndices[i] in 0 until this.rows}) { "Row index exceeded dim 1" }
    require(colIndices.all{ i -> colIndices[i] in 0 until this.columns}) { "Column index exceeded dim 2" }

    this.values = values.copyOf()
    this.rowIndices = rowIndices.copyOf()
    this.colIndices = colIndices.copyOf()

    return this
  }

  /**
   *
   */
  override fun assignValues(a: NDArray<*>): SparseNDArray {

    return when(a) {
      is DenseNDArray -> TODO("not implemented")
      is SparseNDArray -> this.assignValues(a)
      is SparseBinaryNDArray -> TODO("not implemented")
      else -> throw RuntimeException("Invalid NDArray type")
    }
  }

  /**
   *
   */
  override fun assignValues(a: NDArray<*>, mask: NDArrayMask): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun sum(): Double {
    return (0 until this.values.size).sumByDouble { i -> this.values[i] }
  }

  /**
   *
   */
  override fun sum(n: Double): SparseNDArray {
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

    for (index in 0 until this.values.size) {
      this.values[index] += n
    }

    return this
  }

  /**
   *
   */
  override fun assignSum(a: NDArray<*>): SparseNDArray {

    return when(a) {
      is DenseNDArray -> TODO("not implemented")
      is SparseNDArray -> this.assignSum(a)
      is SparseBinaryNDArray -> TODO("not implemented")
      else -> throw RuntimeException("Invalid NDArray type")
    }
  }

  /**
   *
   */
  fun assignSum(a: SparseNDArray): SparseNDArray {
    require(a.shape == this.shape) { "Arrays with different size" }
    require(a.values.size == this.values.size) { "Arrays with a different amount of active values" }

    for (index in 0 until this.values.size) {
      this.values[index] += a.values[index]
    }

    return this
  }

  /**
   *
   */
  fun assignSumMerging(a: SparseNDArray): SparseNDArray {
    require(a.shape == this.shape) { "Arrays with different size" }

    val activeIndices = Array(
      size = this.values.size + a.values.size,
      init = { i ->
        val ref: SparseNDArray = if (i < this.values.size) this else a
        val index: Int = i % this.values.size

        val value: Double = ref.values[index]
        val row: Int = ref.rowIndices[index]
        val col: Int = ref.colIndices[index]

        SparseEntry(Indices(row, col), value)
      })

    val values = arrayListOf<Double>()
    val rows = arrayListOf<Int>()
    val columns = arrayListOf<Int>()

    for ((indices, value) in activeIndices.sortedWith(Comparator<SparseEntry> { (aIndices), (bIndices) ->
      if (aIndices.second != bIndices.second) {
        aIndices.second - bIndices.second
      } else {
        aIndices.first - bIndices.first
      }
    })) {
      if (value != 0.0) {
        if (values.size == 0 || columns.last() != indices.second || rows.last() != indices.first) {
          values.add(value)
          rows.add(indices.first)
          columns.add(indices.second)
        } else {
          values[values.lastIndex] += value
        }
      }
    }

    this.values = values.toTypedArray()
    this.rowIndices = rows.toTypedArray()
    this.colIndices = columns.toTypedArray()

    return this
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
  override fun dot(a: NDArray<*>): SparseNDArray {
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
  fun assignDot(a: DenseNDArray, b: SparseBinaryNDArray): SparseNDArray {
    require(a.rows == this.rows) { "a.rows (%d) != this.rows (%d)".format(a.rows, this.rows) }
    require(b.columns == this.columns) { "b.columns (%d) != this.columns (%d)".format(b.columns, this.columns) }
    require(a.columns == b.rows) { "a.columns (%d) != b.rows (%d)".format(a.columns, b.rows) }

    if (b.rows == 1) {
      // Column vector (dot) row vector
      this.zeros()

      val valuesCount = b.activeIndicesByColumn.keys.size * a.rows
      val values = Array(size = valuesCount, init = { 0.0 })
      val rows = Array(size = valuesCount, init = { 0 })
      val columns = Array(size = valuesCount, init = { 0 })

      var k = 0
      for (j in b.activeIndicesByColumn.keys) {
         for (i in 0 until a.rows) {
           values[k] = a[i]
           rows[k] = i
           columns[k] = j
           k++
         }
      }

      this.values = values
      this.rowIndices = rows
      this.colIndices = columns

    } else if (b.columns == 1) {
      // n-dim array (dot) column vector
      this.zeros()
      this.values = Array(size = a.rows, init = { i -> b.activeIndicesByRow.keys.sumByDouble { a[i, it] } })
      this.rowIndices = Array(size = a.rows, init = { it })
      this.colIndices = Array(size = a.rows, init = { 0 })


    } else {
      // n-dim array (dot) n-dim array
      TODO("not implemented")
    }

    return this
  }

  /**
   *
   */
  override fun assignDot(a: DenseNDArray, b: NDArray<*>): SparseNDArray {

    when(b) {
      is DenseNDArray -> TODO("not implemented")
      is SparseNDArray -> TODO("not implemented")
      is SparseBinaryNDArray -> this.assignDot(a, b)
    }

    return this
  }

  /**
   *
   */
  override fun prod(n: Double): SparseNDArray {

    return SparseNDArray(
      shape = this.shape,
      values = Array(size = this.values.size, init = { i -> this.values[i] * n }),
      rows = this.rowIndices.copyOf(),
      columns = this.colIndices.copyOf())
  }

  /**
   *
   */
  override fun prod(a: NDArray<*>): SparseNDArray {

    return when(a) {
      is DenseNDArray -> this.prod(a)
      is SparseNDArray -> this.prod(a)
      is SparseBinaryNDArray -> TODO("not implemented")
      else -> throw RuntimeException("Invalid NDArray type")
    }
  }

  /**
   *
   */
  private fun prod(a: SparseNDArray): SparseNDArray {
    require(a.shape == this.shape) { "Arrays with different size" }
    require(a.values.size == this.values.size) { "Arrays with a different amount of active values" }

    return SparseNDArray(
      shape = this.shape,
      values = Array(size = this.values.size, init = { i -> this.values[i] * a.values[i]}),
      rows = this.rowIndices.copyOf(),
      columns = this.colIndices.copyOf())
  }

  /**
   *
   */
  private fun prod(a: DenseNDArray): SparseNDArray {
    require(a.shape == this.shape) { "Arrays with different size" }

    return SparseNDArray(
      shape = this.shape,
      values = Array(
        size = this.values.size,
        init = { i -> this.values[i] * a[this.rowIndices[i], this.colIndices[i]]}
      ),
      rows = this.rowIndices.copyOf(),
      columns = this.colIndices.copyOf())
  }

  /**
   *
   */
  override fun prod(n: Double, mask: NDArrayMask): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignProd(n: Double): SparseNDArray {

    for (index in 0 until this.values.size) {
      this.values[index] *= n
    }

    return this
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
    require(a.shape == this.shape) { "Arrays with different size" }
    require(a.values.size == this.values.size) { "Arrays with a different amount of active values" }

    for (index in 0 until this.values.size) {
      this.values[index] *= a.values[index]
    }

    return this
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

    return when(a) {
      is DenseNDArray -> TODO("not implemented")
      is SparseNDArray -> this.div(a)
      is SparseBinaryNDArray -> TODO("not implemented")
      else -> throw RuntimeException("Invalid NDArray type")
    }
  }

  /**
   *
   */
  override fun div(a: SparseNDArray): SparseNDArray {
    require(a.shape == this.shape) { "Arrays with different size" }
    require(a.values.size == this.values.size) { "Arrays with a different amount of active values" }

    return SparseNDArray(
      shape = this.shape,
      values = Array(size = this.values.size, init = { i -> this.values[i] / a.values[i]}),
      rows = this.rowIndices.copyOf(),
      columns = this.colIndices.copyOf())
  }

  /**
   *
   */
  override fun assignDiv(n: Double): SparseNDArray {

    for (index in 0 until this.values.size) {
      this.values[index] /= n
    }

    return this
  }

  /**
   *
   */
  override fun assignDiv(a: SparseNDArray): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun avg(): Double {
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
   * Euclidean norm of this NDArray
   *
   * @return the euclidean norm
   */
  override fun norm2(): Double {
    TODO("not implemented")
  }

  /**
   * @return the index of the maximum value (-1 if empty)
   */
  override fun argMaxIndex(): Int {
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
   *
   */
  override fun randomize(randomGenerator: RandomGenerator): SparseNDArray {
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
   * Splits this NDArray into multiple NDArray each with length [splittingLength]
   *
   * @param splittingLength the length for sub-array division
   *
   * @return an Array containing the split values
   */
  override fun splitV(splittingLength: Int): Array<SparseNDArray> {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun equals(a: SparseNDArray, tolerance: Double): Boolean {

    this.sortValues()
    a.sortValues()

    return equals(this.values, a.values, tolerance = tolerance) &&
      Arrays.equals(this.rowIndices, a.rowIndices) &&
      Arrays.equals(this.colIndices, a.colIndices)
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
    return other is SparseNDArray && this.equals(other)
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
  private fun sortValues() {
    if (this.colIndices.size > 1) {
      this.quicksort(0, this.colIndices.lastIndex)
    }
  }

  /**
   *
   */
  private fun quicksort(lo: Int, hi: Int) {

    if (lo < hi) {
      val p: Int = this.partition(lo, hi)
      this.quicksort(lo, p - 1)
      this.quicksort(p + 1, hi)
    }
  }

  /**
   *
   */
  private fun partition(lo: Int, hi: Int): Int {

    val pivot: Int = hi
    var i: Int = lo

    while (i < pivot && this.compareArrays(i, pivot) <= 0) i++

    for (j in (i + 1) until hi) {
      if (this.compareArrays(j, pivot) <= 0) {
        this.swap(i++, j)
      }
    }

    this.swap(i, pivot)

    return i
  }

  /**
   *
   */
  private fun compareArrays(i: Int, j: Int): Int {
    return if (this.colIndices[i] != this.colIndices[j])
      this.colIndices[i] - this.colIndices[j]
    else
      this.rowIndices[i] - this.rowIndices[j]
  }

  /**
   *
   */
  private fun swap(i: Int, j: Int) {
    if (i != j) {
      this.swapArray(this.values, i, j)
      this.swapArray(this.rowIndices, i, j)
      this.swapArray(this.colIndices, i, j)
    }
  }

  /**
   *
   */
  private fun <T> swapArray(array: Array<T>, i: Int, j: Int) {
    val tmp: T = array[i]
    array[i] = array[j]
    array[j] = tmp
  }
}
