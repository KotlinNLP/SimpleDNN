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
 * The configuration of a merge layer with a variable output size and and an optional activation function.
 *
 * @property type the connection type
 * @property outputSize the size of the merged output
 * @property activationFunction the output activation function
 */
abstract class VariableOutputMergeConfig(
  type: LayerType.Connection,
  val outputSize: Int,
  val activationFunction: ActivationFunction?
) : MergeConfiguration(type)
