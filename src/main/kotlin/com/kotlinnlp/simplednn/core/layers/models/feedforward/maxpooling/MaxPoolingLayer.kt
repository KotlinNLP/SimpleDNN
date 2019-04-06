/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.maxpooling

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.layers.Layer
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.helpers.RelevanceHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The Max Pooling Layer Structure.
 *
 * @property inputArray the input array of the layer
 * @property inputSize the input layer shape
 * @property poolSize The pooling window shape. By default, stride is equal to poolSize.
 * @property params the parameters which connect the input to the output
 * @property activationFunction the activation function of the layer
 * @property dropout the probability of dropout (default 0.0).
 *                   If applying it, the usual value is 0.5 (better 0.25 if it's the first layer).
 * @property id an identification number useful to track a specific layer (default: 0)
 */

class MaxPoolingLayer <InputNDArrayType : NDArray<InputNDArrayType>>(
    inputArray: AugmentedArray<InputNDArrayType>,
    inputType: LayerType.Input,
    val inputSize: Shape,
    val poolSize: Shape,
    override val params: MaxPoolingLayerParameters,
    activationFunction: ActivationFunction? = null,
    dropout: Double = 0.0,
    override val id: Int = 0
) : Layer<InputNDArrayType>(
    inputArray = inputArray,
    inputType = inputType,
    outputArray = AugmentedArray(1),
    params = params,
    activationFunction = activationFunction,
    dropout = dropout
) {

  init {
    require(inputArray.values.shape == inputSize)
    require(inputArray.values.rows % poolSize.dim1 == 0)
    require(inputArray.values.columns % poolSize.dim2 == 0)
  }

  /**
   * Return the [Shape] of output layers
   */
  private fun getOutputShape(): Shape {
    val x: Int = inputArray.values.rows / poolSize.dim1
    val y: Int = inputArray.values.columns / poolSize.dim2
    return Shape(x,y)
  }

  /**
   * The helper which executes the forward
   */
  override val forwardHelper = MaxPoolingForwardHelper(layer = this)

  /**
   * The helper which executes the backward
   */
  override val backwardHelper = MaxPoolingBackwardHelper(layer = this)

  /**
   * The helper which calculates the relevance
   */
  override val relevanceHelper: RelevanceHelper? = null

  /**
   * The output array
   */
  override val outputArray = AugmentedArray(DenseNDArrayFactory.zeros(getOutputShape()))

  /**
   * The matrix containing the argmax row index of output array.
   */
  val argMaxi = Array(getOutputShape().dim1) {IntArray(getOutputShape().dim2)}

  /**
   * The matrix containing the argmax column index of output array.
   */
  val argMaxj = Array(getOutputShape().dim1) {IntArray(getOutputShape().dim2)}

}
