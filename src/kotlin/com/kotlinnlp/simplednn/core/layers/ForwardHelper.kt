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
   * Forward the input to the output combining it with the parameters, saving the contributions.
   *
   * @param layerContributions the structure in which to save the contributions during the calculations
   */
  abstract fun forward(layerContributions: LayerParameters)

  /**
   * Forward [x] to [y] combining it with [w] and [b], saving the contributions in [contributions].
   *
   * @param contributions a matrix which maps the contributions from each value of [x] to each value of [y]
   * @param x a generic [NDArray]
   * @param y a [DenseNDArray]
   * @param w the weights which maps [x] to [y]
   * @param b the biases added to each value of [y] (if null no bias is added)
   */
  protected fun forwardArray(contributions: NDArray<*>,
                             x: NDArray<*>,
                             y: DenseNDArray,
                             w: DenseNDArray,
                             b: DenseNDArray? = null) {

    when (x) {

      is DenseNDArray -> this.forwardDenseArray(
        x = x,
        y = y,
        w = w,
        b = b,
        contributions = contributions as DenseNDArray)

      is SparseBinaryNDArray -> this.forwardSparseBinaryArray(
        x = x,
        y = y,
        w = w,
        b = b,
        contributions = contributions as SparseNDArray)

      else -> throw RuntimeException("Invalid input type '%s'".format(x.javaClass.name))
    }
  }

  /**
   * Forward [x] to [y] combining it with [w] and [b], saving the contributions in [contributions].
   *
   * @param contributions a matrix which maps the contributions from each value of [x] to each value of [y]
   * @param x a [DenseNDArray]
   * @param y a [DenseNDArray]
   * @param w the weights which maps [x] to [y]
   * @param b the biases added to each value of [y] (if null no bias is added)
   */
  protected fun forwardDenseArray(contributions: DenseNDArray,
                                  x: DenseNDArray,
                                  y: DenseNDArray,
                                  w: DenseNDArray,
                                  b: DenseNDArray? = null) {

    val xLength: Int = x.length

    for (j in 0 until w.rows) {

      y[j] = 0.0

      for (i in 0 until w.columns) {
        var contribution: Double = w[j, i] * x[i]

        if (b != null) {
          contribution += b[j] / xLength
        }

        contributions[j, i] = contribution
        y[j] += contribution
      }
    }
  }

  /**
   * Forward [x] to [y] combining it with [w] and [b], saving the contributions in [contributions].
   *
   * @param contributions a matrix which maps the contributions from each value of [x] to each value of [y]
   * @param x the input array of the layer
   * @param y the output array of the layer
   * @param w the weights which connect [x] to [y]
   * @param b the biases added to each value of [y] (if null no bias is added)
   */
  private fun forwardSparseBinaryArray(contributions: SparseNDArray,
                                       x: SparseBinaryNDArray,
                                       y: DenseNDArray,
                                       w: DenseNDArray,
                                       b: DenseNDArray? = null) {

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

        var contribution: Double = w[j, i]

        if (b != null) {
          contribution += b[j] / xActiveIndicesSize // biases are distributed uniformly on the active values
        }

        y[j] += contribution

        contribution
      })

    contributions.assignValues(
      values = values,
      rowIndices = Array(size = contrActiveIndicesSize, init = { k -> k % yLength}),
      colIndices = Array(size = contrActiveIndicesSize, init = { k -> xActiveIndices[k / yLength]}))
  }


  /**
   * Add the recurrent contribution to the output array, saving the contributions of the input in respect of the output.
   *
   * y += wRec (dot) yPrev
   *
   * @param contributions a matrix which maps the contributions from each value of [yPrev] to each value of [yRec]
   * @param yPrev the output array of the layer in the previous state
   * @param yRec the array in which the recurrent contribution is saved
   * @param y the output array of the layer
   * @param wRec the recurrent weights which connect [yPrev] to [y]
   * @param b the biases added to each value of [yRec] (if null no bias is added)
   */
  protected fun addRecurrentContribution(contributions: DenseNDArray,
                                         yPrev: DenseNDArray,
                                         yRec: DenseNDArray,
                                         y: DenseNDArray,
                                         wRec: DenseNDArray,
                                         b: DenseNDArray) {

    this.forwardArray(contributions = contributions, x = yPrev, y = yRec, w = wRec, b = b)

    y.assignSum(yRec)
  }

}
