/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.mergelayers.biaffine

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.deeplearning.mergelayers.MergeLayer
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The Biaffine Layer Structure.
 *
 * @property inputArray the first input array of the layer
 * @property inputArray2 the second input array of the layer
 * @property outputArray the output array of the layer
 * @property params the parameters which connect the input to the output
 * @property activationFunction the activation function of the layer
 * @property dropout the probability of dropout (default 0.0).
 *                   If applying it, the usual value is 0.5 (better 0.25 if it's the first layer).
 */
class BiaffineLayerStructure<InputNDArrayType : NDArray<InputNDArrayType>>(
  inputArray1: AugmentedArray<InputNDArrayType>,
  inputArray2: AugmentedArray<InputNDArrayType>,
  outputArray: AugmentedArray<DenseNDArray>,
  override val params: BiaffineLayerParameters,
  activationFunction: ActivationFunction? = null,
  dropout: Double = 0.0
) : MergeLayer<InputNDArrayType>(
  inputArray1 = inputArray1,
  inputArray2 = inputArray2,
  outputArray = outputArray,
  params = params,
  activationFunction = activationFunction,
  dropout = dropout) {

  /**
   * The helper which execute the forward.
   */
  override val forwardHelper = BiaffineForwardHelper(layer = this)

  /**
   * The helper which execute the backward.
   */
  override val backwardHelper = BiaffineBackwardHelper(layer = this)

  /**
   * The helper which calculates the relevance.
   */
  override val relevanceHelper = BiaffineRelevanceHelper(layer = this)

  /**
   * Initialization: set the activation function of the outputArray.
   */
  init {
    if (activationFunction != null) {
      outputArray.setActivation(activationFunction)
    }
  }

  /**
   * @return the [BiaffineLayerParameters] used to store errors
   */
  override fun parametersErrorsFactory() = BiaffineLayerParameters(
    inputSize1 = this.params.inputSize1,
    inputSize2 = this.params.inputSize2,
    outputSize = this.params.outputSize,
    sparseInput = this.params.sparseInput
  )
}
