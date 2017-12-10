/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.multipredictionscorer

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.layers.LayerType

/**
 * The configuration of a Feedforward Neural Network used in the MultiPredictionScorer.
 *
 * @property inputSize the size of the input layer
 * @property inputType the type of the input arrays (default Dense)
 * @property inputDropout the probability of dropout of the input layer (default 0.0).
 *                        If applying it, the usual value is 0.25.
 * @property hiddenSize the size of the hidden layer
 * @property hiddenActivation the activation function of the hidden layer
 * @property hiddenDropout the probability of dropout of the hidden layer (default 0.0).
 *                         If applying it, the usual value is 0.5.
 * @property outputSize the size of the output layer
 * @property outputActivation the activation function of the output layer
 */
data class MultiPredictionNetworkConfig(
  val inputSize: Int,
  val inputType: LayerType.Input = LayerType.Input.Dense,
  val inputDropout: Double = 0.0,
  val hiddenSize: Int,
  val hiddenActivation: ActivationFunction?,
  val hiddenDropout: Double = 0.0,
  val outputSize: Int,
  val outputActivation: ActivationFunction?
)
