/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.norm

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.Layer
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.helpers.RelevanceHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The normalization layer.
 *
 * Reference:
 * [Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinto, 2016, Layer Normalization](https://arxiv.org/abs/1607.06450)
 *
 * @property inputArray the input array of the layer
 * @property inputType the input array type (default Dense)
 * @property outputArray the output array of the layer
 * @property params the parameters which connect the input to the output
 * @property id an identification number useful to track a specific layer (default: 0)
 */
class NormLayer<InputNDArrayType : NDArray<InputNDArrayType>>(
  inputArray: AugmentedArray<InputNDArrayType>,
  inputType: LayerType.Input,
  outputArray: AugmentedArray<DenseNDArray>,
  override val params: NormLayerParameters,
  override val id: Int = 0
) : Layer<InputNDArrayType>(
  inputArray = inputArray,
  inputType = inputType,
  outputArray = outputArray,
  params = params,
  activationFunction = null,
  dropout = 0.0
) {

  /**
   * Support for the mean of the input array.
   */
  internal var mean: Double = 0.0

  /**
   * Support for the standard deviation of the input array.
   */
  internal var stdDev: Double = 0.0

  /**
   * The helper which executes the forward.
   */
  override val forwardHelper = NormForwardHelper(layer = this)

  /**
   * The helper which executes the backward.
   */
  override val backwardHelper = NormBackwardHelper(layer = this)

  /**
   * The helper which calculates the relevance.
   */
  override val relevanceHelper: RelevanceHelper? = null

  /**
   * Check the size of the input arrays.
   */
  init {

    require(this.inputArray.size == this.outputArray.size) {
      "The input and the output arrays must have the same size."
    }
  }
}
