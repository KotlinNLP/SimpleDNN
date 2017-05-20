/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.birnn

import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters

/**
 * The BiRNNParameters contains the parameter of its sub-networks
 * (leftToRightNetwork, rightToLeftNetwork, outputNetwork)
 *
 * @property leftToRight network parameters of the left-to-right recurrent neural network
 * @property rightToLeft network parameters of the right-to-left recurrent neural network
 * @property output network parameters of the final feed-forward network
 */
data class BiRNNParameters(
  val leftToRight: NetworkParameters,
  val rightToLeft: NetworkParameters,
  val output: NetworkParameters)
