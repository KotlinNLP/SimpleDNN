/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.gru

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.activations.Sigmoid
import com.kotlinnlp.simplednn.core.layers.*
import com.kotlinnlp.simplednn.core.layers.recurrent.*
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * The CRU Layer Structure.
 *
 * @property inputArray the input array of the layer
 * @property outputArray the output array of the layer
 * @property params the parameters which connect the input to the output
 * @property layerContextWindow the context window used for the forward and the backward
 * @property activationFunction the activation function of the layer
 * @property dropout the probability of dropout (default 0.0).
 *                   If applying it, the usual value is 0.5 (better 0.25 if it's the first layer).
 */
class GRULayerStructure<InputNDArrayType : NDArray<InputNDArrayType>>(
  inputArray: AugmentedArray<InputNDArrayType>,
  outputArray: AugmentedArray<DenseNDArray>,
  params: LayerParameters<*>,
  layerContextWindow: LayerContextWindow,
  activationFunction: ActivationFunction? = null,
  dropout: Double = 0.0
) : GatedRecurrentLayerStructure<InputNDArrayType>(
  inputArray = inputArray,
  outputArray = outputArray,
  params = params,
  layerContextWindow = layerContextWindow,
  activationFunction = activationFunction,
  dropout = dropout) {

  /**
   *
   */
  val candidate = RecurrentLayerUnit<InputNDArrayType>(outputArray.size)

  /**
   *
   */
  val resetGate = RecurrentLayerUnit<InputNDArrayType>(outputArray.size)

  /**
   *
   */
  val partitionGate = RecurrentLayerUnit<InputNDArrayType>(outputArray.size)

  /**
   * The helper which executes the forward
   */
  override val forwardHelper = GRUForwardHelper(layer = this)

  /**
   * The helper which executes the backward
   */
  override val backwardHelper = GRUBackwardHelper(layer = this)

  /**
   * The helper which calculates the relevance
   */
  override val relevanceHelper = GRURelevanceHelper(layer = this)

  /**
   * Initialization: set the activation function of the gates
   */
  init {

    this.resetGate.setActivation(Sigmoid())
    this.partitionGate.setActivation(Sigmoid())

    if (activationFunction != null) {
      this.candidate.setActivation(activationFunction)
    }
  }
}
