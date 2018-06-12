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
 * A data class that defines that the output Merge layer of a [com.kotlinnlp.simplednn.deeplearning.birnn.BiRNN] is an
 * Affine layer.
 *
 * @property outputSize the size of the merged output
 * @property activationFunction the output activation function
 * @property dropout the probability of dropout
 */
class AffineMerge(
  outputSize: Int,
  activationFunction: ActivationFunction? = null,
  dropout: Double = 0.0
) : OpenOutputMerge(
  type = LayerType.Connection.Affine,
  dropout = dropout,
  outputSize = outputSize,
  activationFunction = activationFunction
)
