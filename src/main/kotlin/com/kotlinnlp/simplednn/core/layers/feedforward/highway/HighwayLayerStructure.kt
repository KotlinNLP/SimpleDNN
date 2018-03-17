/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.feedforward.highway

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.Sigmoid
import com.kotlinnlp.simplednn.core.layers.*
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.utils.ItemsPool

/**
 * The Highway Layer Structure.
 *
 * @property inputArray the input array of the layer
 * @property outputArray the output array of the layer
 * @property params the parameters which connect the input to the output
 * @property activationFunction the activation function of the layer
 * @property dropout the probability of dropout (default 0.0).
 *                   If applying it, the usual value is 0.5 (better 0.25 if it's the first layer).
 * @property id an identification number useful to track a specific layer (default: 0)
 */
class HighwayLayerStructure<InputNDArrayType : NDArray<InputNDArrayType>>(
  inputArray: AugmentedArray<InputNDArrayType>,
  outputArray: AugmentedArray<DenseNDArray>,
  params: LayerParameters<*>,
  activationFunction: ActivationFunction? = null,
  dropout: Double = 0.0,
  override val id: Int = 0
) : ItemsPool.IDItem,
  LayerStructure<InputNDArrayType>(
    inputArray = inputArray,
    outputArray = outputArray,
    params = params,
    activationFunction = activationFunction,
    dropout = dropout
  ) {

  /**
   * The input unit.
   */
  val inputUnit = LayerUnit<InputNDArrayType>(outputArray.size)

  /**
   * The transform gate.
   */
  val transformGate = LayerUnit<InputNDArrayType>(outputArray.size)

  /**
   * The helper which executes the forward
   */
  override val forwardHelper = HighwayForwardHelper(layer = this)

  /**
   * The helper which executes the backward
   */
  override val backwardHelper = HighwayBackwardHelper(layer = this)

  /**
   * The helper which calculates the relevance
   */
  override val relevanceHelper = HighwayRelevanceHelper(layer = this)

  /**
   * Initialization: set the activation function of the outputArray
   */
  init {

    this.transformGate.setActivation(Sigmoid())

    if (activationFunction != null) {
      this.inputUnit.setActivation(activationFunction)
    }
  }
}
