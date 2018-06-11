/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.merge.affine

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.models.merge.MergeLayer
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The Affine Layer Structure.
 *
 * @property inputArrays the input arrays of the layer
 * @property outputArray the output array of the layer
 * @property params the parameters which connect the input to the output
 * @property activationFunction the activation function of the layer
 * @property dropout the probability of dropout (default 0.0).
 *                   If applying it, the usual value is 0.5 (better 0.25 if it's the first layer).
 * @property id an identification number useful to track a specific [AffineLayerStructure]
 */
class AffineLayerStructure<InputNDArrayType : NDArray<InputNDArrayType>>(
  inputArrays: List<AugmentedArray<InputNDArrayType>>,
  outputArray: AugmentedArray<DenseNDArray>,
  override val params: AffineLayerParameters,
  activationFunction: ActivationFunction? = null,
  dropout: Double = 0.0,
  id: Int = 0
) : MergeLayer<InputNDArrayType>(
  inputArrays = inputArrays,
  outputArray = outputArray,
  params = params,
  activationFunction = activationFunction,
  dropout = dropout,
  id = id) {

  init { this.checkInputSize() }

  /**
   * The helper which execute the forward.
   */
  override val forwardHelper = AffineForwardHelper(layer = this)

  /**
   * The helper which execute the backward.
   */
  override val backwardHelper = AffineBackwardHelper(layer = this)

  /**
   * The helper which calculates the relevance.
   */
  override val relevanceHelper = AffineRelevanceHelper(layer = this)

  /**
   * Initialization: set the activation function of the outputArray.
   */
  init {
    if (activationFunction != null) {
      outputArray.setActivation(activationFunction)
    }
  }

  /**
   * @return the [AffineLayerParameters] used to store errors
   */
  override fun parametersErrorsFactory() = AffineLayerParameters(
    inputsSize = this.params.inputsSize,
    outputSize = this.params.outputSize,
    sparseInput = this.params.sparseInput,
    weightsInitializer = null,
    biasesInitializer = null
  )
}
