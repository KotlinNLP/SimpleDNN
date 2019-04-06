/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.convolution

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.*
import com.kotlinnlp.simplednn.core.layers.helpers.RelevanceHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The Convolution Layer Structure.
 *
 * @property inputArrays the input arrays of the layer
 * @property inputType the input array type (default Dense)
 * @property outputArrays the output arrays of the layer
 * @property inputSize the input layer shape
 * @property xStride The stride length of the convolution on x-axis
 * @property yStride The stride length of the convolution on y-axis
 * @property params the parameters which connect the input to the output
 * @property activationFunction the activation function of the layer
 * @property dropout the probability of dropout (default 0.0).
 *                   If applying it, the usual value is 0.5 (better 0.25 if it's the first layer).
 * @property id an identification number useful to track a specific layer (default: 0)
 */

class ConvolutionLayer<InputNDArrayType : NDArray<InputNDArrayType>>(
    val inputArrays: List<AugmentedArray<InputNDArrayType>>,
    inputType: LayerType.Input,
    val inputSize: Shape,
    val xStride: Int = 1,
    val yStride: Int = 1,
    override val params: ConvolutionLayerParameters,
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

  init {
    require((inputArray.values.rows - inputSize.dim1) % xStride == 0)
    require((inputArray.values.columns - inputSize.dim2) % yStride == 0)
  }

  /**
   * Return the [Shape] of output layers
   */
  private fun getOutputShape(): Shape {
    val x: Int = (inputArray.values.rows - params.kernelSize.dim1) / xStride + 1
    val y: Int = (inputArray.values.columns - params.kernelSize.dim2) / yStride + 1
    return Shape(x, y)
  }

  /**
   * The list containing the layer output
   */
  val outputArrays: List<AugmentedArray<DenseNDArray>> = List(this.params.outputChannels) {
    AugmentedArray(DenseNDArrayFactory.zeros(getOutputShape()))
  }

  /**
   * The helper which executes the forward
   */
  override val forwardHelper = ConvolutionForwardHelper(layer = this)

  /**
   * The helper which executes the backward
   */
  override val backwardHelper = ConvolutionBackwardHelper(layer = this)

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
    require(this.params.outputChannels == outputArrays.size)
    require(this.params.inputChannels == inputArrays.size)
    require(isAllEqualSize(inputArrays))
    require(this.params.kernelSize.dim1 <= inputArrays.first().values.rows)
    require(this.params.kernelSize.dim2 <= inputArrays.first().values.columns)

    if (activationFunction != null) {
      for (outputArray in outputArrays)
        outputArray.setActivation(activationFunction)
    }
  }
}
