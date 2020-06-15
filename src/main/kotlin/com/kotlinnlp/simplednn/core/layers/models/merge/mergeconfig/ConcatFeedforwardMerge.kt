/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.merge.mergeconfig

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.layers.LayerType

/**
 * The ConcatFeedforward merge layer configuration.
 *
 * @property outputSize the size of the merged output
 * @property activationFunction the output activation function
 */
class ConcatFeedforwardMerge(
  outputSize: Int,
  activationFunction: ActivationFunction? = null
) : VariableOutputMergeConfig(
  type = LayerType.Connection.ConcatFeedforward,
  outputSize = outputSize,
  activationFunction = activationFunction
)
