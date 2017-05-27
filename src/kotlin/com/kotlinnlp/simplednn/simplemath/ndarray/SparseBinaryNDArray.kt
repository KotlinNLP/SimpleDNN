/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.simplemath.ndarray

import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator

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
  val activeIndicesByRow: MutableMap<Int, MutableSet<Int>?> = mutableMapOf<Int, MutableSet<Int>?>(),
  val activeIndicesByColumn: MutableMap<Int, MutableSet<Int>?> = mutableMapOf<Int, MutableSet<Int>?>()
) : NDArray<SparseBinaryNDArray>,
    Iterable<Pair<Int, Int>> {

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
  inner class LinearIndicesIterator: Iterator<Pair<Int, Int>> {

    /**
     * The iterator of the map entries by row index (rowIndex, rowActiveIndices)
     */
    val rowsIterator = this@SparseBinaryNDArray.activeIndicesByRow.iterator()

    /**
     * The iterator of the map entries by column index (columnIndex, columnActiveIndices)
     */
    val columnsIterator = this@SparseBinaryNDArray.activeIndicesByColumn.iterator()

    /**
     * The map entry (rowIndex, rowActiveIndices) of the current row
     */
    var curRow: MutableMap.MutableEntry<Int, MutableSet<Int>?>? =
      if (this.rowsIterator.hasNext()) this.rowsIterator.next() else null

    /**
     * The map entry (rowIndex, rowActiveIndices) of the current column
     */
    var curColumn: MutableMap.MutableEntry<Int, MutableSet<Int>?>? =
      if (this.columnsIterator.hasNext()) this.columnsIterator.next() else null

    /**
     *
     */
    var curRowIterator: Iterator<Int>? = this.curRow?.value?.iterator()

    /**
     *
     */
    var curColumnIterator: Iterator<Int>? = this.curColumn?.value?.iterator()

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
    override fun next(): Pair<Int, Int> {

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
  override fun iterator(): Iterator<Pair<Int, Int>> {
    return LinearIndicesIterator()
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
  override val rows: Int = this.shape.dim1

  /**
   *
   */
  override val columns: Int = this.shape.dim2

  /**
   *
   */
  override val isOneHotEncoder: Boolean
    get() = TODO("not implemented") //To change initializer of created properties use File | Settings | File Templates.

  /**
   * Transpose
   */
  override val T: SparseBinaryNDArray get() = SparseBinaryNDArray(
    activeIndicesByRow = this.copyIndices(this.activeIndicesByColumn),
    activeIndicesByColumn = this.copyIndices(this.activeIndicesByRow),
    shape = this.shape.inverse
  )

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
      this.addElement(activeIndicesByRow, key = i, element = j)
      this.addElement(activeIndicesByColumn, key = j, element = i)

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
  private fun addElement(indicesMap: MutableMap<Int, MutableSet<Int>?>, key: Int, element: Int) {

    if (indicesMap.containsKey(key)) {
      // Key already existing
      if (indicesMap[key] != null) {
        indicesMap[key]!!.add(element)
      } else {
        indicesMap[key] = mutableSetOf(0, element)
      }

    } else {
      // New key
      if (element == 0) {
        indicesMap[key] = null
      } else {
        indicesMap[key] = mutableSetOf(element)
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
  private fun copyIndices(indicesMap: MutableMap<Int, MutableSet<Int>?>): MutableMap<Int, MutableSet<Int>?> {

    val newMap = mutableMapOf<Int, MutableSet<Int>?>()

    for ((key, indicesSet) in indicesMap.iterator()) {
      newMap[key] = indicesSet?.toMutableSet()
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
  override fun assignValues(a: NDArray<*>): SparseBinaryNDArray {

    when(a) {
      is DenseNDArray -> TODO("not implemented")
      is SparseNDArray -> {
        this.activeIndicesByRow.clear()
        this.activeIndicesByRow
      }
      is SparseBinaryNDArray -> TODO("not implemented")
      else -> throw RuntimeException("Invalid NDArray type")
    }

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
  override fun div(a: NDArray<*>, mask: NDArrayMask): SparseNDArray {
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
