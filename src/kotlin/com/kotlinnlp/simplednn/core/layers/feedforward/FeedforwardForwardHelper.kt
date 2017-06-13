/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.feedforward

import com.kotlinnlp.simplednn.core.layers.ForwardHelper
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray

/**
 * The helper which executes the forward on a [layer].
 *
 * @property layer the [FeedforwardLayerStructure] in which the forward is executed
 */
class FeedforwardForwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: FeedforwardLayerStructure<InputNDArrayType>
) : ForwardHelper<InputNDArrayType> {

  /**
   * Forward the input to the output combining it with the parameters.
   *
   * y = f(w (dot) x + b)
   */
  override fun forward() { this.layer.params as FeedforwardLayerParameters

    val x: InputNDArrayType = this.layer.inputArray.values
    val y: DenseNDArray = this.layer.outputArray.values

    val w: DenseNDArray = this.layer.params.weights.values as DenseNDArray
    val b: DenseNDArray = this.layer.params.biases.values

    y.assignDot(w, x).assignSum(b)

    this.layer.outputArray.activate()
  }

  /**
   * Forward the input to the output combining it with the parameters, saving the contributes of the parameters.
   *
   * @param paramsContributes the [LayerParameters] in which to save the contributes of the parameters
   */
  override fun forward(paramsContributes: LayerParameters) {
    this.layer.params as FeedforwardLayerParameters
    paramsContributes as FeedforwardLayerParameters

    val x: InputNDArrayType = this.layer.inputArray.values
    val y: DenseNDArray = this.layer.outputArray.values

    val w: DenseNDArray = this.layer.params.weights.values as DenseNDArray
    val b: DenseNDArray = this.layer.params.biases.values

    val wContr: NDArray<*> = paramsContributes.weights.values

    when (x) {
      is DenseNDArray -> this.forwardDenseInput(x = x, y = y, w = w, b = b, contr = wContr as DenseNDArray)
      is SparseBinaryNDArray -> this.forwardSparseInput(x = x, y = y, w = w, b = b, contr = wContr as SparseNDArray)
      else -> throw RuntimeException("Invalid input type '%s'".format(x.javaClass.name))
    }

    this.layer.outputArray.activate()
  }

  /**
   *
   */
  private fun forwardDenseInput(x: DenseNDArray,
                                y: DenseNDArray,
                                w: DenseNDArray,
                                b: DenseNDArray,
                                contr: DenseNDArray) {

    val xLength: Int = x.length

    for (j in 0 until w.rows) {

      y[j] = 0.0

      for (i in 0 until w.columns) {
        val contribute: Double = w[j, i] * x[i] + b[j] / xLength

        contr[j, i] = contribute
        y[j] += contribute
      }
    }
  }

  /**
   *
   */
  private fun forwardSparseInput(x: SparseBinaryNDArray,
                                 y: DenseNDArray,
                                 w: DenseNDArray,
                                 b: DenseNDArray,
                                 contr: SparseNDArray) {

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

    contr.assignValues(
      values = values,
      rowIndices = Array(size = contrActiveIndicesSize, init = { k -> k % yLength}),
      colIndices = Array(size = contrActiveIndicesSize, init = { k -> xActiveIndices[k / yLength]}))
  }
}
