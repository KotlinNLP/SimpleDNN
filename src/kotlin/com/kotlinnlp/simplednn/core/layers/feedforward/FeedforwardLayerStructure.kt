/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.feedforward

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.arrays.Norm1Array
import com.kotlinnlp.simplednn.core.layers.LayerStructure
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray

/**
 * The Feedforward Layer Structure.
 *
 * @property inputArray the input array of the layer
 * @property outputArray the output array of the layer
 * @property params the parameters which connect the input to the output
 * @property activationFunction the activation function of the layer
 * @property dropout the probability of dropout (default 0.0).
 *                   If applying it, the usual value is 0.5 (better 0.25 if it's the first layer).
 */
class FeedforwardLayerStructure<InputNDArrayType : NDArray<InputNDArrayType>>(
  inputArray: AugmentedArray<InputNDArrayType>,
  outputArray: AugmentedArray<DenseNDArray>,
  params: LayerParameters,
  activationFunction: ActivationFunction? = null,
  dropout: Double = 0.0
) : LayerStructure<InputNDArrayType>(
  inputArray = inputArray,
  outputArray = outputArray,
  params = params,
  activationFunction = activationFunction,
  dropout = dropout) {

  /**
   * A support variable to manage the errors on the parameters during the backward
   */
  var paramsErrors: FeedforwardLayerParameters? = null

  /**
   * Initialization: set the activation function to the outputArray
   */
  init {
    if (activationFunction != null) {
      outputArray.setActivation(activationFunction)
    }
  }

  /**
   * Forward the input to the output combining it with the parameters
   *
   * y = f(w (dot) x + b)
   */
  override fun forwardInput() { this.params as FeedforwardLayerParameters

    val x: InputNDArrayType = this.inputArray.values
    val y: DenseNDArray = this.outputArray.values

    val w: DenseNDArray = this.params.weights.values as DenseNDArray
    val b: DenseNDArray = this.params.biases.values

    y.assignDot(w, x).assignSum(b)

    this.outputArray.activate()
  }

  /**
   * Forward the input to the output combining it with the parameters, saving the contributes of the parameters.
   *
   * @param paramsContributes the [LayerParameters] in which to save the contributes of the parameters
   */
  override fun forwardInput(paramsContributes: LayerParameters) {
    this.params as FeedforwardLayerParameters
    paramsContributes as FeedforwardLayerParameters

    val x: InputNDArrayType = this.inputArray.values
    val y: DenseNDArray = this.outputArray.values

    val w: DenseNDArray = this.params.weights.values as DenseNDArray
    val b: DenseNDArray = this.params.biases.values

    val wContr: NDArray<*> = paramsContributes.weights.values

    when (x) {
      is DenseNDArray -> this.forwardDenseInput(x = x, y = y, w = w, b = b, wContr = wContr as DenseNDArray)
      is SparseBinaryNDArray -> this.forwardSparseInput(x = x, y = y, w = w, b = b, wContr = wContr as SparseNDArray)
      else -> throw RuntimeException("Invalid input type '%s'".format(x.javaClass.name))
    }

    this.outputArray.activate()
  }

  /**
   *
   */
  private fun forwardDenseInput(x: DenseNDArray,
                                y: DenseNDArray,
                                w: DenseNDArray,
                                b: DenseNDArray,
                                wContr: DenseNDArray) {

    for (j in 0 until w.rows) {

      y[j] = 0.0

      for (i in 0 until w.columns) {
        val contribute: Double = w[j, i] * x[i] + b[j] / x.length

        wContr[j, i] = contribute
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
                                 wContr: SparseNDArray) {

    y.zeros()

    val xActiveIndices: ArrayList<Int> = x.activeIndicesByColumn.values.first()!!
    val xActiveIndicesSize: Int = xActiveIndices.size
    val yLength: Int = y.length
    val wContrActiveIndicesSize: Int = xActiveIndicesSize * yLength

    val values = Array(
      size = wContrActiveIndicesSize,
      init = { k ->
        val j: Int = k % yLength // linear indexing
        val i: Int = xActiveIndices[k / yLength] // linear indexing

        val contribute: Double = w[j, i] + b[j] / x.length

        y[j] += contribute

        contribute
      })

    wContr.assignValues(
      values = values,
      rowIndices = Array(size = wContrActiveIndicesSize, init = { k -> k / xActiveIndicesSize}),
      colIndices = Array(size = wContrActiveIndicesSize, init = { k -> k % xActiveIndicesSize}))
  }

  /**
   * Calculate the relevance of the input.
   *
   * @param paramsContributes the contributes of the parameters during the last forward
   */
  override fun calculateRelevance(paramsContributes: LayerParameters) {
    this.params as FeedforwardLayerParameters
    paramsContributes as FeedforwardLayerParameters

    val x: InputNDArrayType = this.inputArray.values
    val y: DenseNDArray = this.outputArray.valuesNotActivated
    val yRelevance: DenseNDArray = this.outputArray.relevance.values as DenseNDArray
    // output relevance is always Dense (no Sparse needed, generally little size)

    val wContr: NDArray<*> = paramsContributes.weights.values

    when (x) {

      is DenseNDArray -> this.calculateRelevanceOfDenseInput(
        x = x,
        y = y,
        yRelevance = yRelevance,
        wContr = wContr as DenseNDArray)

      is SparseBinaryNDArray -> this.calculateRelevanceOfSparseInput(
        x = x,
        y = y,
        yRelevance = yRelevance,
        wContr = wContr as SparseNDArray)

      else -> throw RuntimeException("Invalid input type '%s'".format(x.javaClass.name))
    }
  }

  /**
   *
   */
  private fun calculateRelevanceOfDenseInput(x: DenseNDArray,
                                             y: DenseNDArray,
                                             yRelevance: DenseNDArray,
                                             wContr: DenseNDArray) {

    val relevanceArray: DenseNDArray = DenseNDArrayFactory.zeros(shape = Shape(this.inputArray.size))

    for (i in 0 until x.length) {

      for (j in 0 until y.length) {
        val eps: Double = if (y[j] >= 0) this.relevanceEps else -this.relevanceEps
        val epsN: Double = eps / x.length

        relevanceArray[i] += yRelevance[j] * (wContr[j, i]  + epsN) / (y[j] + eps)
      }
    }

    this.inputArray.assignRelevance(Norm1Array(values = relevanceArray))
  }

  /**
   *
   */
  private fun calculateRelevanceOfSparseInput(x: SparseBinaryNDArray,
                                              y: DenseNDArray,
                                              yRelevance: DenseNDArray,
                                              wContr: SparseNDArray) {

    val xActiveIndices: List<Int> = x.activeIndicesByColumn.values.first()!!
    val xActiveIndicesSize: Int =  xActiveIndices.size
    val relevanceValues: Array<Double> = Array(size = xActiveIndicesSize, init = { 0.0 })
    val relevanceColumns: Array<Int> = Array(size = xActiveIndicesSize, init = { 0 })
    val relevanceRows: Array<Int> = xActiveIndices.toTypedArray()
    var k: Int = 0

    for (l in 0 until xActiveIndicesSize) {
      // the indices of the non-zero elements of x are the same of the non-zero columns of wContr
      for (j in 0 until y.length) {
        // loop over the i-th column of wContr (which is non-zero)
        val eps: Double = if (y[j] >= 0) this.relevanceEps else -this.relevanceEps
        val epsN: Double = eps / xActiveIndicesSize
        val wContrIJ: Double = wContr.values[k++]  // linear indexing

        relevanceValues[l] += yRelevance[j] * (wContrIJ + epsN) / (y[j] + eps)
      }
    }

    this.inputArray.assignRelevance(Norm1Array(values = SparseNDArray(
      shape = x.shape,
      values = relevanceValues,
      rows = relevanceRows,
      columns = relevanceColumns
    )))
  }

  /**
   *
   */
  override fun backward(paramsErrors: LayerParameters, propagateToInput: Boolean) {

    this.paramsErrors = paramsErrors as FeedforwardLayerParameters

    this.assignParamsGradients()

    if (propagateToInput) {
      this.assignLayerGradients()
    }
  }

  /**
   * gb = gy * 1
   * gw = gy (dot) x
   */
  private fun assignParamsGradients() {

    val gb: DenseNDArray = this.paramsErrors!!.biases.values
    val gw: NDArray<*> = this.paramsErrors!!.weights.values

    val x: InputNDArrayType = this.inputArray.values
    val gy: DenseNDArray = this.outputArray.errors

    gb.assignValues(gy)
    gw.assignDot(gy, x.T)
  }

  /**
   * gx = (gy (dot) w) * xDeriv
   */
  private fun assignLayerGradients() { this.params as FeedforwardLayerParameters

    val gy: DenseNDArray = this.outputArray.errors
    val w: DenseNDArray = this.params.weights.values as DenseNDArray

    val gx: DenseNDArray = this.inputArray.errors

    gx.assignValues(gy.T.dot(w))

    if (this.inputArray.hasActivation && gx is DenseNDArray) {
      gx.assignProd(this.inputArray.calculateActivationDeriv())
    }
  }
}
