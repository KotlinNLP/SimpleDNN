/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.reshape

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
 * @property inputSize the input shape
 * @property outputSize the output shape
 * @property params the parameters which connect the input to the output
 * @property activationFunction the activation function of the layer
 * @property dropout the probability of dropout (default 0.0).
 *                   If applying it, the usual value is 0.5 (better 0.25 if it's the first layer).
 * @property id an identification number useful to track a specific layer (default: 0)
 */
class ReshapeLayer<InputNDArrayType : NDArray<InputNDArrayType>>(
    inputArray: AugmentedArray<InputNDArrayType>,
    inputType: LayerType.Input,
    val inputSize: Shape,
    val outputSize: Shape,
    override val params: ReshapeLayerParameters,
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
    require(inputSize.dim1 * inputSize.dim2 == outputSize.dim1 * outputSize.dim2)
  }

  /**
   * The helper which executes the forward
   */
  override val forwardHelper = ReshapeForwardHelper(layer = this)

  /**
   * The helper which executes the backward
   */
  override val backwardHelper = ReshapeBackwardHelper(layer = this)

  /**
   * The helper which calculates the relevance
   */
  override val relevanceHelper: RelevanceHelper? = null

  /**
   * The output array
   */
  override val outputArray = AugmentedArray(DenseNDArrayFactory.zeros(outputSize))
}