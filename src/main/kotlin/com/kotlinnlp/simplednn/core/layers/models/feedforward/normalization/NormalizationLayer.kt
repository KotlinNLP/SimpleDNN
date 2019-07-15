/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.normalization

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.layers.Layer
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.helpers.RelevanceHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

class NormalizationLayer <InputNDArrayType : NDArray<InputNDArrayType>>(
    val inputArrays: List<AugmentedArray<InputNDArrayType>>,
    inputType: LayerType.Input,
    val inputSize: Int,
    override val params: NormalizationLayerParameters,
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
   * The list containing the layer output
   */
  val outputArrays: List<AugmentedArray<DenseNDArray>> = List(this.inputArrays.size) {
    AugmentedArray(DenseNDArrayFactory.zeros(Shape(inputSize)))
  }

  /**
   * The vector containing the mean of the input layers
   */
  val meanArray: DenseNDArray = DenseNDArrayFactory.zeros(Shape(inputSize))

  /**
   * The vector containing the standard deviation of the input layers
   */
  val devStdArray: DenseNDArray = DenseNDArrayFactory.zeros(Shape(inputSize))

  /**
   * The helper which executes the forward
   */
  override val forwardHelper = NormalizationForwardHelper(layer = this)

  /**
   * The helper which executes the backward
   */
  override val backwardHelper = NormalizationBackwardHelper(layer = this)

  /**
   * The helper which calculates the relevance
   */
  override val relevanceHelper: RelevanceHelper? = null

  /**
   * Set the values of the inputArray at the given [index].
   *
   * @param index the index of the inputArray to set
   * @param values the values to set into the inputArray1
   */
  fun setInput(index: Int, values: InputNDArrayType) = this.inputArrays[index].assignValues(values)

  /**
   * Set the values of the inputArray at the given [index].
   *
   * @param index the index of the inputArray to set
   * @param values the values to set into the inputArray1
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
   * Check if [arrayList] contains arrays of the same size and shape
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

  /**
   * Initialization: set the activation function of the outputArrays
   */
  init {
    require(isAllEqualSize(inputArrays))

    if (activationFunction != null) {
      for (outputArray in outputArrays)
        outputArray.setActivation(activationFunction)
    }
  }
}