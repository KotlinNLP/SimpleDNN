/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.birnn.mergeconfig

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.layers.LayerType

/**
 * A class that defines the configuration of the output Merge layer of a
 * [com.kotlinnlp.simplednn.deeplearning.birnn.BiRNN] that has a configurable output size and activation.
 *
 * @property type the connection type of the output Merge layer
 * @property dropout the probability of dropout
 * @property outputSize the size of the merged output
 * @property activationFunction the output activation function
 */
abstract class OpenOutputMerge(
  type: LayerType.Connection,
  dropout: Double,
  val outputSize: Int,
  val activationFunction: ActivationFunction?
) : MergeConfiguration(type = type, dropout = dropout)
