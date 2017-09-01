/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.simple

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.layers.*
import com.kotlinnlp.simplednn.core.layers.recurrent.*
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * The SimpleRecurrent Layer Structure.
 *
 * @property inputArray the input array of the layer
 * @property outputArray the output array of the layer
 * @property params the parameters which connect the input to the output
 * @property layerContextWindow the context window used for the forward and the backward
 * @property activationFunction the activation function of the layer
 * @property dropout the probability of dropout (default 0.0).
 *                   If applying it, the usual value is 0.5 (better 0.25 if it's the first layer).
 */
class SimpleRecurrentLayerStructure<InputNDArrayType : NDArray<InputNDArrayType>>(
  inputArray: AugmentedArray<InputNDArrayType>,
  override val outputArray: RecurrentLayerUnit<InputNDArrayType>,
  params: LayerParameters,
  layerContextWindow: LayerContextWindow,
  activationFunction: ActivationFunction? = null,
  dropout: Double = 0.0
) : RecurrentLayerStructure<InputNDArrayType>(
  inputArray = inputArray,
  outputArray = outputArray,
  params = params,
  layerContextWindow = layerContextWindow,
  activationFunction = activationFunction,
  dropout = dropout) {

  /**
   * The helper which executes the forward
   */
  override val forwardHelper = SimpleRecurrentForwardHelper(layer = this)

  /**
   * The helper which executes the backward
   */
  override val backwardHelper = SimpleRecurrentBackwardHelper(layer = this)

  /**
   * The helper which calculates the relevance
   */
  override val relevanceHelper = SimpleRecurrentRelevanceHelper(layer = this)

  /**
   * Initialization: set the activation function of the output array.
   */
  init {
    if (activationFunction != null) {
      outputArray.setActivation(activationFunction)
    }
  }
}
