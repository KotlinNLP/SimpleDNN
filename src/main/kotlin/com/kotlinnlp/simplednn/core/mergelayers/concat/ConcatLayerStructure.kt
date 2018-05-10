/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.mergelayers.concat

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.mergelayers.MergeLayer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The Concat Layer Structure.
 *
 * @property inputArray the first input array of the layer
 * @property inputArray2 the second input array of the layer
 * @property params the parameters which connect the input to the output
 * @property id an identification number useful to track a specific [ConcatLayerStructure]
 */
class ConcatLayerStructure(
  inputArray1: AugmentedArray<DenseNDArray>,
  inputArray2: AugmentedArray<DenseNDArray>,
  outputArray: AugmentedArray<DenseNDArray>,
  override val params: ConcatLayerParameters,
  id: Int = 0
) : MergeLayer<DenseNDArray>(
  inputArray1 = inputArray1,
  inputArray2 = inputArray2,
  outputArray = outputArray,
  params = params,
  activationFunction = null,
  dropout = 0.0,
  id = id) {

  /**
   * The helper which execute the forward.
   */
  override val forwardHelper = ConcatForwardHelper(layer = this)

  /**
   * The helper which execute the backward.
   */
  override val backwardHelper = ConcatBackwardHelper(layer = this)

  /**
   * The helper which calculates the relevance.
   */
  override val relevanceHelper = ConcatRelevanceHelper(layer = this)

  /**
   * @return the [ConcatLayerParameters] used to store errors
   */
  override fun parametersErrorsFactory() = ConcatLayerParameters(
    inputSize1 = this.params.inputSize1,
    inputSize2 = this.params.inputSize2)
}
