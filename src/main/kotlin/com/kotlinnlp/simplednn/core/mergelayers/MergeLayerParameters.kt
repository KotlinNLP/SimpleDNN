/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.mergelayers

import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerParameters

/**
 * The parameters of a merge layer.
 * It has two inputs instead of one.
 *
 * @property inputSize1 the size of the first input
 * @property inputSize2 the size of the second input
 * @property outputSize the size of the output
 * @param weightsInitializer the initializer of the weights (zeros if null)
 * @param biasesInitializer the initializer of the biases (zeros if null)
 * @property sparseInput whether the weights connected to the input are sparse or not
 */
abstract class MergeLayerParameters<SelfType: MergeLayerParameters<SelfType>>(
  val inputSize1: Int,
  val inputSize2: Int,
  outputSize: Int,
  weightsInitializer: Initializer?,
  biasesInitializer: Initializer?,
  val sparseInput: Boolean
) : LayerParameters<SelfType>(
  inputSize = inputSize1,
  outputSize = outputSize,
  weightsInitializer = weightsInitializer,
  biasesInitializer = biasesInitializer
)
