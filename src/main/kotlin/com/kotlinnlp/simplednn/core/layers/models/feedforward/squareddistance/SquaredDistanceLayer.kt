/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.squareddistance

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.Layer
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.helpers.RelevanceHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The Squared Layer Structure.
 *
 * @property inputArray the input array of the layer
 * @property inputType the input array type (default Dense)
 * @property outputArray the output array of the layer
 * @property params the parameters which connect the input to the output
 * @property dropout the probability of dropout
 */
internal class SquaredDistanceLayer<InputNDArrayType : NDArray<InputNDArrayType>>(
  inputArray: AugmentedArray<InputNDArrayType>,
  inputType: LayerType.Input,
  outputArray: AugmentedArray<DenseNDArray>,
  override val params: SquaredDistanceLayerParameters,
  dropout: Double
) : Layer<InputNDArrayType>(
  inputArray = inputArray,
  inputType = inputType,
  outputArray = outputArray,
  params = params,
  activationFunction = null,
  dropout = dropout
) {

  /**
   * It is a support variable used during the calculations.
   */
  val bhOut: AugmentedArray<DenseNDArray> = AugmentedArray(this.params.rank)

  /**
   * The helper which executes the forward
   */
  override val forwardHelper = SquaredDistanceForwardHelper(layer = this)

  /**
   * The helper which executes the backward
   */
  override val backwardHelper = SquaredDistanceBackwardHelper(layer = this)

  /**
   * The helper which calculates the relevance
   */
  override val relevanceHelper: RelevanceHelper? = null
}
