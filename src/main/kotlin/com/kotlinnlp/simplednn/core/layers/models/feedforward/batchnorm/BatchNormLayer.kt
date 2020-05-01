/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.batchnorm

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
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
 * Reference:
 * [Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinto, 2016, Layer Normalization](https://arxiv.org/abs/1607.06450)
 *
 * @property inputArrays the input arrays of the layer
 * @property inputType the type of the input arrays
 * @property params the parameters which connect the input to the output
 * @property id an identification number useful to track a specific layer (default: 0)
 */
class BatchNormLayer<InputNDArrayType : NDArray<InputNDArrayType>>(
  val inputArrays: List<AugmentedArray<InputNDArrayType>>,
  inputType: LayerType.Input,
  override val params: BatchNormLayerParameters,
  override val id: Int = 0
) : Layer<InputNDArrayType>(
  inputArray = inputArrays[0],
  inputType = inputType,
  outputArray = AugmentedArray(1),
  params = params,
  activationFunction = null,
  dropout = 0.0
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
   * Support vector for the mean of the input arrays.
   */
  internal val mean: DenseNDArray = DenseNDArrayFactory.zeros(Shape(this.inputSize))

  /**
   * Support vector for the standard deviation of the input arrays.
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
   * Check the size of the input arrays.
   */
  init {

    require(this.inputArrays.all { it.size == this.inputSize }) {
      "All the input arrays must have the same size."
    }
  }

  /**
   * Set the values of the input arrays.
   *
   * @param inputs the values of the input arrays
   */
  fun setInputs(inputs: List<InputNDArrayType>) {

    this.inputArrays.zip(inputs).forEach { (array, values) ->
      array.assignValues(values)
    }
  }

  /**
   * Set the errors of the output arrays.
   *
   * @param outputErrors the errors of each output array
   */
  fun setErrors(outputErrors: List<DenseNDArray>) {

    this.outputArrays.zip(outputErrors).forEach { (array, error) ->
      array.assignErrors(error)
    }
  }

  /**
   * @param copy whether the returned errors must be a copy or a reference
   *
   * @return the errors of the input arrays
   */
  fun getInputErrors(copy: Boolean = true): List<DenseNDArray> = this.inputArrays.map {
    if (copy) it.errors.copy() else it.errors
  }
}
