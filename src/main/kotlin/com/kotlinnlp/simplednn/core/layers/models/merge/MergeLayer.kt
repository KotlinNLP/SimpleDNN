/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.merge

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.layers.Layer
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The Merge Layer abstract class.
 * It is a [Layer] with two inputs instead of one.
 *
 * @property inputArrays the input arrays of the layer
 * @property outputArray the output array of the layer
 * @property params the parameters which connect the input to the output
 * @property activationFunction the activation function of the layer
 * @property dropout the probability of dropout (default 0.0).
 *                   If applying it, the usual value is 0.5 (better 0.25 if it's the first layer).
 * @property id an identification number useful to track a specific [MergeLayer]
 */
abstract class MergeLayer<InputNDArrayType : NDArray<InputNDArrayType>>(
  val inputArrays: List<AugmentedArray<InputNDArrayType>>,
  outputArray: AugmentedArray<DenseNDArray>,
  override val params: MergeLayerParameters<*>,
  activationFunction: ActivationFunction? = null,
  dropout: Double = 0.0,
  id: Int = 0
) : Layer<InputNDArrayType>(
  inputArray = inputArrays[0],
  outputArray = outputArray,
  params = params,
  activationFunction = activationFunction,
  dropout = dropout,
  id = id) {

  /**
   * Set the values of the inputArray at the given [index].
   *
   * @param index the index of the inputArray to set
   * @param values the values to set into the inputArray1
   */
  fun setInput(index: Int, values: InputNDArrayType) = this.inputArrays[index].assignValues(values)

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return a list containing the errors of each input array
   */
  fun getInputErrors(copy: Boolean = true): List<DenseNDArray> = this.inputArrays.map {
    if (copy) it.errors.copy() else it.errors
  }

  /**
   * Ensure that the input arrays are compatible with the parameters.
   */
  protected fun checkInputSize() {

    require(this.inputArrays.size > 1)
    require(this.inputArrays.size == this.params.inputsSize.size)
    require(this.inputArrays.zip(this.params.inputsSize).all { it.first.size == it.second })
  }
}
