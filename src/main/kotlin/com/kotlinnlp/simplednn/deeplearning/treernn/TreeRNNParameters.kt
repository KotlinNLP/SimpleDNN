/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.treernn

import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters

/**
 * The TreeRNNParameters contains the parameter of its sub-networks
 * (leftRNN, rightRNN, concatNetwork)
 *
 * @property leftRNN network parameters of the left recurrent neural network
 * @property rightRNN network parameters of the right recurrent neural network
 * @property concatNetwork network parameters of the final feed-forward network
 */
data class TreeRNNParameters(
  val leftRNN: NetworkParameters,
  val rightRNN: NetworkParameters,
  val concatNetwork: NetworkParameters)
