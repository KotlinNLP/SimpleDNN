/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.batchnorm

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.layers.Layer
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.helpers.RelevanceHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The batch normalization layer structure.
 *
 * @property inputArrays the input arrays of the layer
 * @property inputType the type of the input arrays
 * @property params the parameters which connect the input to the output
 * @property activationFunction the activation function of the layer
 * @property dropout the probability of dropout (default 0.0).
 *                   If applying it, the usual value is 0.5 (better 0.25 if it's the first layer).
 * @property id an identification number useful to track a specific layer (default: 0)
 */
class BatchNormLayer<InputNDArrayType : NDArray<InputNDArrayType>>(
  val inputArrays: List<AugmentedArray<InputNDArrayType>>,
  inputType: LayerType.Input,
  override val params: BatchNormLayerParameters,
  activationFunction: ActivationFunction? = null,
  dropout: Double = 0.0,
  override val id: Int = 0
) : Layer<InputNDArrayType>(
  inputArray = inputArrays[0],
  inputType = inputType,
  outputArray = AugmentedArray(1),
  params = params,
  activationFunction = activationFunction,
  dropout = dropout
) {

  /**
   * The size of the input arrays.
   */
  val inputSize: Int = this.inputArrays.first().values.length

  /**
   * The list containing the layer output.
   */
  val outputArrays: List<AugmentedArray<DenseNDArray>> = List(this.inputArrays.size) {
    AugmentedArray(DenseNDArrayFactory.zeros(Shape(this.inputSize)))
  }

  /**
   * The vector containing the mean of the input arrays.
   */
  internal val mean: DenseNDArray = DenseNDArrayFactory.zeros(Shape(this.inputSize))

  /**
   * The vector containing the standard deviation of the input arrays.
   */
  internal val stdDev: DenseNDArray = DenseNDArrayFactory.zeros(Shape(this.inputSize))

  /**
   * The helper which executes the forward.
   */
  override val forwardHelper = BatchNormForwardHelper(layer = this)

  /**
   * The helper which executes the backward.
   */
  override val backwardHelper = BatchNormBackwardHelper(layer = this)

  /**
   * The helper which calculates the relevance.
   */
  override val relevanceHelper: RelevanceHelper? = null

  /**
   * Check sizes and set activation functions.
   */
  init {

    require(this.inputArrays.all { it.size == this.inputSize }) {
      "All the input arrays must have the same size."
    }

    require(isAllEqualSize(this.inputArrays))

    if (activationFunction != null)
      this.outputArrays.forEach { it.setActivation(activationFunction) }
  }

  /**
   * Set the values of the input array at a given index.
   *
   * @param index the index of an input array
   * @param values the values to set
   */
  fun setInput(index: Int, values: InputNDArrayType) = this.inputArrays[index].assignValues(values)

  /**
   * Set the errors of the output array at a given index.
   *
   * @param index the index of the output array
   * @param errors the errors to set
   */
  fun setErrors(index: Int, errors: DenseNDArray) = this.outputArrays[index].assignErrors(errors)

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return a list containing the errors of each input array
   */
  fun getInputErrors(copy: Boolean = true): List<DenseNDArray> = this.inputArrays.map {
    if (copy) it.errors.copy() else it.errors
  }

  /**
   * Perform the multiplication of the output arrays by the derivative of its activated values.
   */
  fun applyOutputActivationDerivs() {

    for (array in this.outputArrays)
      if (array.hasActivation) {
        array.errors.assignProd(array.calculateActivationDeriv())
      }
  }

  /**
   * @param arrayList a list of augmented arrays
   *
   * @return `true` if all the arrays in [arrayList] have the same size and shape, otherwise `false`
   */
  private fun isAllEqualSize(arrayList: List<AugmentedArray<InputNDArrayType>>): Boolean {

    val columns = arrayList.first().values.columns
    val rows = arrayList.first().values.rows

    arrayList.forEach { array ->
      if (array.values.columns != columns || array.values.rows != rows)
        return false
    }

    return true
  }
}
