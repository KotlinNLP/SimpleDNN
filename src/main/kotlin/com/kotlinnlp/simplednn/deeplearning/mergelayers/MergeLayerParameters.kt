/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.mergelayers

import com.kotlinnlp.simplednn.core.layers.LayerParameters

/**
 * The parameters of a merge layer.
 * It has two inputs instead of one.
 *
 * @property inputSize1 the size of the first input
 * @property inputSize2 the size of the second input
 * @property outputSize the size of the output
 * @property sparseInput whether the weights connected to the input are sparse or not
 */
abstract class MergeLayerParameters(
  val inputSize1: Int,
  val inputSize2: Int,
  outputSize: Int,
  val sparseInput: Boolean
) : LayerParameters<MergeLayerParameters>(
  inputSize = inputSize1,
  outputSize = outputSize
)
