/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers

import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray

/**
 * The helper which calculates the relevance of the input of a [layer] respect of its output.
 *
 * @property layer the [LayerStructure] in which to calculate the input relevance
 */
abstract class RelevanceHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  protected val layer: LayerStructure<InputNDArrayType>
) {

  /**
   * The stabilizing term used to calculate the relevance
   */
  private val relevanceEps: Double = 0.01

  /**
   * Calculate the relevance of the input.
   *
   * @param paramsContributes the contributes of the parameters during the last forward
   */
  abstract fun calculateRelevance(paramsContributes: LayerParameters)

  /**
   * Calculate the relevance of the array [x] respect of the calculation which produced the array [y].
   *
   * @param x a generic [NDArray]
   * @param y a [DenseNDArray] (no Sparse needed, generally little size on output)
   * @param yRelevance a [DenseNDArray], whose norm is 1.0, which indicates how much relevant are the values of [y]
   * @param contributes a matrix which maps the contributes from each value of [x] to each value of [y]
   *
   * @return the relevance of [x] respect of [y]
   */
  protected fun calculateRelevanceOfArray(x: InputNDArrayType,
                                          y: DenseNDArray,
                                          yRelevance: DenseNDArray,
                                          contributes: NDArray<*>): NDArray<*> =
    when (x) {

      is DenseNDArray -> this.calculateRelevanceOfDenseArray(
        x = x,
        y = y,
        yRelevance = yRelevance,
        contributes = contributes as DenseNDArray)

      is SparseBinaryNDArray -> this.calculateRelevanceOfSparseArray(
        x = x,
        y = y,
        yRelevance = yRelevance,
        contributes = contributes as SparseNDArray)

      else -> throw RuntimeException("Invalid input type '%s'".format(x.javaClass.name))
    }

  /**
   * Calculate the relevance of the Dense array [x] respect of the calculation which produced the Dense array [y].
   *
   * @param x a [DenseNDArray]
   * @param y a [DenseNDArray] (no Sparse needed, generally little size on output)
   * @param yRelevance a [DenseNDArray], whose norm is 1.0, which indicates how much relevant are the values of [y]
   * @param contributes a matrix which contains the contributes of each value of [x] to calculate each value of [y]
   *
   * @return the relevance of [x] respect of [y]
   */
  protected fun calculateRelevanceOfDenseArray(x: DenseNDArray,
                                               y: DenseNDArray,
                                               yRelevance: DenseNDArray,
                                               contributes: DenseNDArray): DenseNDArray {

    val relevanceArray: DenseNDArray = DenseNDArrayFactory.zeros(shape = Shape(this.layer.inputArray.size))
    val xLength: Int = x.length
    val yLength: Int = y.length

    for (i in 0 until xLength) {

      for (j in 0 until yLength) {
        val eps: Double = if (y[j] >= 0) this.relevanceEps else -this.relevanceEps
        val epsN: Double = eps / xLength

        relevanceArray[i] += yRelevance[j] * (contributes[j, i]  + epsN) / (y[j] + eps)
      }
    }

    return relevanceArray
  }

  /**
   * Calculate the relevance of the SparseBinary array [x] respect of the calculation which produced the Dense array
   * [y].
   *
   * @param x a [SparseBinaryNDArray]
   * @param y a [DenseNDArray] (no Sparse needed, generally little size on output)
   * @param yRelevance a [DenseNDArray], whose norm is 1.0, which indicates how much relevant are the values of [y]
   * @param contributes a matrix which contains the contributes of each value of [x] to calculate each value of [y]
   *
   * @return the relevance of [x] respect of [y]
   */
  private fun calculateRelevanceOfSparseArray(x: SparseBinaryNDArray,
                                              y: DenseNDArray,
                                              yRelevance: DenseNDArray,
                                              contributes: SparseNDArray): SparseNDArray {

    val xActiveIndices: List<Int> = x.activeIndicesByColumn.values.first()!!
    val xActiveIndicesSize: Int =  xActiveIndices.size
    val relevanceValues: Array<Double> = Array(size = xActiveIndicesSize, init = { 0.0 })
    val relevanceColumns: Array<Int> = Array(size = xActiveIndicesSize, init = { 0 })
    val relevanceRows: Array<Int> = xActiveIndices.toTypedArray()
    val yLength: Int = y.length
    var k: Int = 0

    for (l in 0 until xActiveIndicesSize) {
      // the indices of the non-zero elements of x are the same of the non-zero columns of contributes
      for (j in 0 until yLength) {
        // loop over the i-th column of contributes (which is non-zero)
        val eps: Double = if (y[j] >= 0) this.relevanceEps else -this.relevanceEps
        val epsN: Double = eps / xActiveIndicesSize
        val wContrJI: Double = contributes.values[k++]  // linear indexing

        relevanceValues[l] += yRelevance[j] * (wContrJI + epsN) / (y[j] + eps)
      }
    }

    return SparseNDArray(
      shape = x.shape,
      values = relevanceValues,
      rows = relevanceRows,
      columns = relevanceColumns
    )
  }
}
