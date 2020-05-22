/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.simple

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.*
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.utils.ItemsPool

/**
 * The Feedforward Layer Structure.
 *
 * @property inputArray the input array of the layer
 * @property inputType the input array type (default Dense)
 * @property outputArray the output array of the layer
 * @property params the parameters which connect the input to the output
 * @property activationFunction the activation function of the layer
 * @property dropout the probability of dropout (default 0.0).
 *                   If applying it, the usual value is 0.5 (better 0.25 if it's the first layer).
 * @property id an identification number useful to track a specific layer (default: 0)
 */
internal class FeedforwardLayer<InputNDArrayType : NDArray<InputNDArrayType>>(
  inputArray: AugmentedArray<InputNDArrayType>,
  inputType: LayerType.Input,
  outputArray: AugmentedArray<DenseNDArray>,
  override val params: FeedforwardLayerParameters,
  activationFunction: ActivationFunction? = null,
  dropout: Double = 0.0,
  override val id: Int = 0
) : ItemsPool.IDItem,
  Layer<InputNDArrayType>(
    inputArray = inputArray,
    inputType = inputType,
    outputArray = outputArray,
    params = params,
    activationFunction = activationFunction,
    dropout = dropout
  ) {

  /**
   * The helper which executes the forward
   */
  override val forwardHelper = FeedforwardForwardHelper(layer = this)

  /**
   * The helper which executes the backward
   */
  override val backwardHelper = FeedforwardBackwardHelper(layer = this)

  /**
   * The helper which calculates the relevance
   */
  @Suppress("UNCHECKED_CAST")
  override val relevanceHelper: FeedforwardRelevanceHelper? = if (this.denseInput)
    FeedforwardRelevanceHelper(layer = this as FeedforwardLayer<DenseNDArray>)
  else
    null

  /**
   * Initialization: set the activation function of the outputArray
   */
  init {
    if (activationFunction != null) {
      this.outputArray.setActivation(activationFunction)
    }
  }
}
