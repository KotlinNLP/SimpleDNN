/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.merge.concatff

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.helpers.RelevanceHelper
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayer
import com.kotlinnlp.simplednn.core.layers.models.merge.MergeLayer
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The Concat Feedforward Layer Structure.
 *
 * @property inputArrays the input arrays of the layer
 * @property inputType the input array type (default Dense)
 * @property outputArray the output array of the layer
 * @property params the parameters which connect the input to the output
 * @property activationFunction the activation function of the output
 * @property dropout the probability of dropout
 */
internal class ConcatFFLayer<InputNDArrayType : NDArray<InputNDArrayType>>(
  inputArrays: List<AugmentedArray<InputNDArrayType>>,
  inputType: LayerType.Input,
  outputArray: AugmentedArray<DenseNDArray>,
  override val params: ConcatFFLayerParameters,
  activationFunction: ActivationFunction? = null,
  dropout: Double
) : MergeLayer<InputNDArrayType>(
  inputArrays = inputArrays,
  inputType = inputType,
  outputArray = outputArray,
  params = params,
  activationFunction = null,
  dropout = dropout
) {

  /**
   * The helper which execute the forward.
   */
  override val forwardHelper = ConcatFFForwardHelper(layer = this)

  /**
   * The helper which execute the backward.
   */
  override val backwardHelper = ConcatFFBackwardHelper(layer = this)

  /**
   * The helper which calculates the relevance
   */
  override val relevanceHelper: RelevanceHelper? = null

  /**
   * The output feed-forward layer.
   * TODO: make it working also with non-dense input arrays.
   */
  internal val outputFeedforward = FeedforwardLayer<DenseNDArray>(
    inputArray = AugmentedArray(size = this.params.inputsSize.sum()),
    inputType = inputType,
    outputArray = outputArray,
    params = this.params.output,
    dropout = dropout,
    activationFunction = activationFunction
  ).apply {
    setParamsErrorsCollector(this@ConcatFFLayer.getParamsErrorsCollector())
  }

  init {
    this.checkInputSize()
  }
}
