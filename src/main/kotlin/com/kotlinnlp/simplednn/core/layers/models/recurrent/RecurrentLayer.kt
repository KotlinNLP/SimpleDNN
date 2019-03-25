/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.recurrent

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.layers.*
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import java.lang.RuntimeException

/**
 * The Recurrent Layer Structure.
 *
 * @property inputArray the input array of the layer
 * @property inputType the input array type (default Dense)
 * @property outputArray the output array of the layer
 * @property params the parameters which connect the input to the output
 * @property layerContextWindow the context window used for the forward and the backward
 * @property activationFunction the activation function of the layer
 * @property dropout the probability of dropout (default 0.0).
 *                   If applying it, the usual value is 0.5 (better 0.25 if it's the first layer).
 */
abstract class RecurrentLayer<InputNDArrayType : NDArray<InputNDArrayType>>(
  inputArray: AugmentedArray<InputNDArrayType>,
  inputType: LayerType.Input,
  outputArray: AugmentedArray<DenseNDArray>,
  params: LayerParameters<*>,
  val layerContextWindow: LayerContextWindow,
  activationFunction: ActivationFunction? = null,
  dropout: Double = 0.0
) : Layer<InputNDArrayType>(
  inputArray = inputArray,
  inputType = inputType,
  outputArray = outputArray,
  params = params,
  activationFunction = activationFunction,
  dropout = dropout) {

  /**
   * The helper which calculates the relevance
   */
  abstract override val relevanceHelper: RecurrentRelevanceHelper?

  /**
   * Calculate the relevance of the output in the previous state respect of the current one and assign it to the output
   * array of the previous state.
   *
   * @param layerContributions the contributions saved during the last forward
   */
  fun setRecurrentRelevance(layerContributions: LayerParameters<*>) {
    this.relevanceHelper?.setRecurrentRelevance(layerContributions = layerContributions) ?:
      throw RuntimeException("Relevance propagation not available.")
  }

  /**
   * Set the initial hidden array.
   * This method should be used when this layer is used as initial hidden state in a recurrent neural network.
   *
   * @param array the initial hidden array
   */
  abstract fun setInitHidden(array: DenseNDArray)

  /**
   * Get the errors of the initial hidden array.
   * This method should be used only if this layer is used as initial hidden state in a recurrent neural network.
   *
   * @return the errors of the initial hidden array
   */
  abstract fun getInitHiddenErrors(): DenseNDArray
}
