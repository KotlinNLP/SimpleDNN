/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers

import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray

/**
 * The helper which executes the forward on a [layer].
 *
 * @property layer the [LayerStructure] in which the forward is executed.
 */
abstract class ForwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  open protected val layer: LayerStructure<InputNDArrayType>
) {

  /**
   * Forward the input to the output combining it with the parameters.
   */
  abstract fun forward()

  /**
   * Forward the input to the output combining it with the parameters, saving the contributes of the parameters.
   *
   * @param paramsContributes the [LayerParameters] in which to save the contributes of the parameters
   */
  abstract fun forward(paramsContributes: LayerParameters)

  /**
   * Forward [x] to [y] combining it with [w] and [b], saving the contributes in [contributes].
   *
   * @param x a generic [NDArray]
   * @param y a [DenseNDArray]
   * @param w the weights which maps [x] to [y]
   * @param b the biases added to each value of [y]
   * @param contributes a matrix which maps the contributes from each value of [x] to each value of [y]
   */
  protected fun forwardArray(x: NDArray<*>,
                             y: DenseNDArray,
                             w: DenseNDArray,
                             b: DenseNDArray,
                             contributes: NDArray<*>) {

    when (x) {

      is DenseNDArray -> this.forwardDenseArray(
        x = x,
        y = y,
        w = w,
        b = b,
        contributes = contributes as DenseNDArray)

      is SparseBinaryNDArray -> this.forwardSparseBinaryArray(
        x = x,
        y = y,
        w = w,
        b = b,
        contributes = contributes as SparseNDArray)

      else -> throw RuntimeException("Invalid input type '%s'".format(x.javaClass.name))
    }
  }

  /**
   * Forward [x] to [y] combining it with [w] and [b], saving the contributes in [contributes].
   *
   * @param x a [DenseNDArray]
   * @param y a [DenseNDArray]
   * @param w the weights which maps [x] to [y]
   * @param b the biases added to each value of [y]
   * @param contributes a matrix which maps the contributes from each value of [x] to each value of [y]
   */
  private fun forwardDenseArray(x: DenseNDArray,
                                y: DenseNDArray,
                                w: DenseNDArray,
                                b: DenseNDArray,
                                contributes: DenseNDArray) {

    val xLength: Int = x.length

    for (j in 0 until w.rows) {

      y[j] = 0.0

      for (i in 0 until w.columns) {
        val contribute: Double = w[j, i] * x[i] + b[j] / xLength

        contributes[j, i] = contribute
        y[j] += contribute
      }
    }
  }

  /**
   * Forward [x] to [y] combining it with [w] and [b], saving the contributes in [contributes].
   *
   * @param x a [SparseBinaryNDArray]
   * @param y a [DenseNDArray]
   * @param w the weights which maps [x] to [y]
   * @param b the biases added to each value of [y]
   * @param contributes a matrix which maps the contributes from each value of [x] to each value of [y]
   */
  private fun forwardSparseBinaryArray(x: SparseBinaryNDArray,
                                       y: DenseNDArray,
                                       w: DenseNDArray,
                                       b: DenseNDArray,
                                       contributes: SparseNDArray) {

    val xActiveIndices: ArrayList<Int> = x.activeIndicesByColumn.values.first()!!
    val xActiveIndicesSize: Int = xActiveIndices.size
    val yLength: Int = y.length
    val contrActiveIndicesSize: Int = xActiveIndicesSize * yLength

    y.zeros()

    val values = Array(
      size = contrActiveIndicesSize,
      init = { k ->
        val j: Int = k % yLength // linear indexing
        val i: Int = xActiveIndices[k / yLength] // linear indexing

        val biasN: Double = b[j] / xActiveIndicesSize // biases are distributed uniformly on the active values
        val contribute: Double = w[j, i] + biasN

        y[j] += contribute

        contribute
      })

    contributes.assignValues(
      values = values,
      rowIndices = Array(size = contrActiveIndicesSize, init = { k -> k % yLength}),
      colIndices = Array(size = contrActiveIndicesSize, init = { k -> xActiveIndices[k / yLength]}))
  }
}
