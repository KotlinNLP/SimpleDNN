/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.feedforward

import com.kotlinnlp.simplednn.core.arrays.Norm1Array
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.RelevanceHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray

/**
 * The helper which calculates the relevance of the input of a [layer] respect of its output.
 *
 * @property layer the [FeedforwardLayerStructure] in which to calculate the input relevance
 */
class FeedforwardRelevanceHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: FeedforwardLayerStructure<InputNDArrayType>
) : RelevanceHelper<InputNDArrayType> {

  /**
   * The stabilizing term used to calculate the relevance
   */
  private val relevanceEps: Double = 0.01

  /**
   * Calculate the relevance of the input.
   *
   * @param paramsContributes the contributes of the parameters during the last forward
   */
  override fun calculateRelevance(paramsContributes: LayerParameters) {
    this.layer.params as FeedforwardLayerParameters
    paramsContributes as FeedforwardLayerParameters

    val x: InputNDArrayType = this.layer.inputArray.values
    val y: DenseNDArray = this.layer.outputArray.valuesNotActivated
    val yRelevance: DenseNDArray = this.layer.outputArray.relevance.values as DenseNDArray
    // output relevance is always Dense (no Sparse needed, generally little size)

    val contr: NDArray<*> = paramsContributes.weights.values

    when (x) {

      is DenseNDArray -> this.calculateRelevanceOfDenseInput(
        x = x,
        y = y,
        yRelevance = yRelevance,
        contr = contr as DenseNDArray)

      is SparseBinaryNDArray -> this.calculateRelevanceOfSparseInput(
        x = x,
        y = y,
        yRelevance = yRelevance,
        contr = contr as SparseNDArray)

      else -> throw RuntimeException("Invalid input type '%s'".format(x.javaClass.name))
    }
  }

  /**
   *
   */
  private fun calculateRelevanceOfDenseInput(x: DenseNDArray,
                                             y: DenseNDArray,
                                             yRelevance: DenseNDArray,
                                             contr: DenseNDArray) {

    val relevanceArray: DenseNDArray = DenseNDArrayFactory.zeros(shape = Shape(this.layer.inputArray.size))
    val xLength: Int = x.length
    val yLength: Int = y.length

    for (i in 0 until xLength) {

      for (j in 0 until yLength) {
        val eps: Double = if (y[j] >= 0) this.relevanceEps else -this.relevanceEps
        val epsN: Double = eps / xLength

        relevanceArray[i] += yRelevance[j] * (contr[j, i]  + epsN) / (y[j] + eps)
      }
    }

    this.layer.inputArray.assignRelevance(Norm1Array(values = relevanceArray))
  }

  /**
   *
   */
  private fun calculateRelevanceOfSparseInput(x: SparseBinaryNDArray,
                                              y: DenseNDArray,
                                              yRelevance: DenseNDArray,
                                              contr: SparseNDArray) {

    val xActiveIndices: List<Int> = x.activeIndicesByColumn.values.first()!!
    val xActiveIndicesSize: Int =  xActiveIndices.size
    val relevanceValues: Array<Double> = Array(size = xActiveIndicesSize, init = { 0.0 })
    val relevanceColumns: Array<Int> = Array(size = xActiveIndicesSize, init = { 0 })
    val relevanceRows: Array<Int> = xActiveIndices.toTypedArray()
    val yLength: Int = y.length
    var k: Int = 0

    for (l in 0 until xActiveIndicesSize) {
      // the indices of the non-zero elements of x are the same of the non-zero columns of contr
      for (j in 0 until yLength) {
        // loop over the i-th column of contr (which is non-zero)
        val eps: Double = if (y[j] >= 0) this.relevanceEps else -this.relevanceEps
        val epsN: Double = eps / xActiveIndicesSize
        val wContrJI: Double = contr.values[k++]  // linear indexing

        relevanceValues[l] += yRelevance[j] * (wContrJI + epsN) / (y[j] + eps)
      }
    }

    this.layer.inputArray.assignRelevance(Norm1Array(values = SparseNDArray(
      shape = x.shape,
      values = relevanceValues,
      rows = relevanceRows,
      columns = relevanceColumns
    )))
  }
}
