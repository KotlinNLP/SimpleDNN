/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary

import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.simplemath.ndarray.*
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray

/**
 * An [NDArray] of sparse binary values.
 *
 * @property shape the shape of the array
 * @property activeIndicesByRow a [MutableMap] which maps a row index with the set of active indices of that row
 *                              (the set is null if the only active index in the row is 0)
 * @property activeIndicesByColumn a [MutableMap] which maps a column index with the set of active indices of that
 *                                 column (the set is null if the only active index in the column is 0)
 */
class SparseBinaryNDArray(
  override val shape: Shape,
  val activeIndicesByRow: VectorsMap = mutableMapOf(),
  val activeIndicesByColumn: VectorsMap = mutableMapOf()
) : NDArray<SparseBinaryNDArray>,
    Iterable<Indices> {

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
  inner class LinearIndicesIterator: Iterator<Indices> {

    /**
     * The iterator of the map entries by row index (rowIndex, rowActiveIndices)
     */
    private val rowsIterator = this@SparseBinaryNDArray.activeIndicesByRow.toSortedMap().iterator()

    /**
     * The iterator of the map entries by column index (columnIndex, columnActiveIndices)
     */
    private val columnsIterator = this@SparseBinaryNDArray.activeIndicesByColumn.toSortedMap().iterator()

    /**
     * The map entry (rowIndex, rowActiveIndices) of the current row
     */
    private var curRow: VectorsMapEntry? = if (this.rowsIterator.hasNext()) this.rowsIterator.next() else null

    /**
     * The map entry (rowIndex, rowActiveIndices) of the current column
     */
    private var curColumn: VectorsMapEntry? = if (this.columnsIterator.hasNext()) this.columnsIterator.next() else null

    /**
     *
     */
    private var curRowIterator: Iterator<Int>? = this.curRow?.value?.iterator()

    /**
     *
     */
    private var curColumnIterator: Iterator<Int>? = this.curColumn?.value?.iterator()

    /**
     *
     */
    override fun hasNext(): Boolean {

      return if (this@SparseBinaryNDArray.rows == 1)
        this.curRowIterator != null && this.curRowIterator!!.hasNext()
      else
        this.columnsIterator.hasNext() || (this.curColumnIterator != null && this.curColumnIterator!!.hasNext())
    }

    /**
     *
     */
    override fun next(): Indices {

      return if (this@SparseBinaryNDArray.rows == 1) {

        Pair(this.curRow!!.key, this.curRowIterator!!.next())

      } else {

        this.updateColumnIterator()

        Pair(this.curColumnIterator!!.next(), this.curColumn!!.key)
      }
    }

    /**
     *
     */
    private fun updateColumnIterator() {

      if (this@SparseBinaryNDArray.columns > 1 && !this.curColumnIterator!!.hasNext()) {
        this.curColumn = this.columnsIterator.next()
        this.curColumnIterator = this.curColumn!!.value!!.iterator()
      }
    }
  }

  /**
   * Iterator over active indices
   */
  override fun iterator(): Iterator<Indices> = LinearIndicesIterator()

  /**
   *
   */
  override val factory = SparseBinaryNDArrayFactory

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
  override val t: SparseBinaryNDArray get() = SparseBinaryNDArray(
    activeIndicesByRow = this.copyIndices(this.activeIndicesByColumn),
    activeIndicesByColumn = this.copyIndices(this.activeIndicesByRow),
    shape = this.shape.inverse
  )

  /**
   * The mask representing the active indices of this [SparseBinaryNDArray].
   */
  val mask: NDArrayMask get() = this.buildMask()

  /**
   * @return the mask representing the active indices of this [SparseBinaryNDArray]
   */
  private fun buildMask(): NDArrayMask {
    val rowIndices = mutableListOf<Int>()
    val colIndices = mutableListOf<Int>()

    this.activeIndicesByColumn.forEach { j, column ->
      if (column != null)
        column.forEach { i -> rowIndices.add(i); colIndices.add(j) }
      else {
        rowIndices.add(0); colIndices.add(j)
      }
    }

    return NDArrayMask(dim1 = rowIndices.toIntArray(), dim2 = colIndices.toIntArray())
  }

  /**
   *
   */
  override fun get(i: Int): Int {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun get(i: Int, j: Int): Int {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun set(i: Int, value: Number) {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun set(i: Int, j: Int, value: Number) {
    require(value is Int && (value == 0 || value == 1) && i < this.rows && j < this.columns)

    if (value == 1) {
      this.addElement(this.activeIndicesByRow, key = i, element = j)
      this.addElement(this.activeIndicesByColumn, key = j, element = i)

    } else {
      TODO("not implemented")
    }
  }

  /**
   *
   */
  fun set(i: Int, j: Int) = this.set(i, j, 1)

  /**
   *
   */
  private fun addElement(indicesMap: VectorsMap, key: Int, element: Int, sortElements: Boolean = true) {

    if (indicesMap.containsKey(key)) {
      // Key already existing
      if (indicesMap[key] != null) {
        if (!indicesMap[key]!!.contains(element)) {
          indicesMap[key]!!.add(element)
          if (sortElements) indicesMap[key]!!.sort()
        }
      } else {
        indicesMap[key] = arrayListOf(0, element)
      }

    } else {
      // New key
      if (element == 0) {
        indicesMap[key] = null
      } else {
        indicesMap[key] = arrayListOf(element)
      }
    }
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
   * Get a one-dimensional NDArray sub-vector of a vertical vector.
   *
   * @param a the start index of the range (inclusive)
   * @param b the end index of the range (exclusive)
   *
   * @return the sub-array
   */
  override fun getRange(a: Int, b: Int): SparseBinaryNDArray {
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
  override fun zerosLike(): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun copy(): SparseBinaryNDArray = SparseBinaryNDArray(
    activeIndicesByRow = this.copyIndices(this.activeIndicesByRow),
    activeIndicesByColumn = this.copyIndices(this.activeIndicesByColumn),
    shape = this.shape.copy()
  )

  /**
   *
   */
  private fun copyIndices(indicesMap: VectorsMap): VectorsMap {

    val newMap = mutableMapOf<Int, VectorIndices?>()

    for ((key, indicesList) in indicesMap.iterator()) {
      newMap[key] = indicesList?.toMutableList()
    }

    return newMap
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
  override fun assignValues(a: NDArray<*>): SparseBinaryNDArray = when(a) {
    is DenseNDArray -> TODO("not implemented")
    is SparseNDArray -> TODO("not implemented")
    is SparseBinaryNDArray -> this.assignValues(a)
    else -> throw RuntimeException("Invalid NDArray type")
  }

  /**
   *
   */
  private fun assignValues(a: SparseBinaryNDArray): SparseBinaryNDArray {

    this.activeIndicesByRow.clear()
    this.activeIndicesByColumn.clear()

    for ((i, j) in a) {
      this.addElement(this.activeIndicesByRow, key = i, element = j, sortElements = false)
      this.addElement(this.activeIndicesByColumn, key = j, element = i, sortElements = false)
    }

    this.activeIndicesByRow.forEach { _, u -> u?.sort() }
    this.activeIndicesByColumn.forEach { _, u -> u?.sort() }

    return this
  }

  /**
   *
   */
  override fun assignValues(a: NDArray<*>, mask: NDArrayMask): SparseBinaryNDArray {
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
  override fun sum(n: Double): SparseBinaryNDArray {
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
  override fun assignSub(a: NDArray<*>): SparseBinaryNDArray {
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
  override fun dot(a: NDArray<*>): DenseNDArray {
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
  override fun prod(a: NDArray<*>): SparseBinaryNDArray {
    TODO("not implemented")
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
  override fun assignProd(n: Double): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignProd(n: Double, mask: NDArrayMask): SparseBinaryNDArray {
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
  override fun div(a: NDArray<*>): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun div(a: SparseNDArray): SparseNDArray {
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
   *
   */
  override fun avg(): Double {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun abs(): SparseBinaryNDArray {
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
  override fun sqrt(): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   * Square root of this [SparseBinaryNDArray] masked by [mask]
   *
   * @param mask the mask to apply
   *
   * @return a [SparseNDArray]
   */
  override fun sqrt(mask: NDArrayMask): SparseNDArray {
    TODO("not implemented")
  }

  /**
   * Power.
   *
   * @param power the exponent
   *
   * @return a new [SparseBinaryNDArray] containing the values of this to the power of [power]
   */
  override fun pow(power: Double): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   * In-place power.
   *
   * @param power the exponent
   *
   * @return this [SparseBinaryNDArray] to the power of [power]
   */
  override fun assignPow(power: Double): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   * Logarithm with base 10.
   *
   * @return a new [SparseBinaryNDArray] containing the element-wise logarithm with base 10 of this array
   */
  override fun log10(): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   * In-place logarithm with base 10.
   *
   * @return this [SparseBinaryNDArray] after having applied the logarithm with base 10 to its values
   */
  override fun assignLog10(): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   * Natural logarithm.
   *
   * @return a new [SparseBinaryNDArray] containing the element-wise natural logarithm of this array
   */
  override fun ln(): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   * In-place logarithm with base 10.
   *
   * @return this [SparseBinaryNDArray] after having applied the logarithm with base 10 to its values
   */
  override fun assignLn(): SparseBinaryNDArray {
    TODO("not implemented")
  }

  /**
   * The norm (L1 distance) of this NDArray.
   *
   * @return the norm
   */
  override fun norm(): Double {
    TODO("not implemented")
  }

  /**
   * The Euclidean norm of this DenseNDArray.
   *
   * @return the euclidean norm
   */
  override fun norm2(): Double {
    TODO("not implemented")
  }

  /**
   * Get the index of the highest value eventually skipping the element at the given [exceptIndex] when it is >= 0.
   *
   * @param exceptIndex the index to exclude
   *
   * @return the index of the maximum value (-1 if empty)
   */
  override fun argMaxIndex(exceptIndex: Int): Int {
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
   *
   */
  override fun randomize(randomGenerator: RandomGenerator): SparseBinaryNDArray {
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
   * Splits this NDArray into more NDArrays.
   *
   * If the number of arguments is one, split this NDArray into multiple NDArray each with length [splittingLength].
   * If there are multiple arguments, split this NDArray according to the length of each [splittingLength] element.
   *
   * @param splittingLength the length(s) for sub-array division
   *
   * @return a list containing the split values
   */
  override fun splitV(vararg splittingLength: Int): List<SparseBinaryNDArray> {
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
}
