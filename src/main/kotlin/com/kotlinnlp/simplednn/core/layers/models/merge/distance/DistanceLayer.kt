/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.merge.distance

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.layers.models.merge.MergeLayer
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.utils.ItemsPool

class DistanceLayer<InputNDArrayType : NDArray<InputNDArrayType>>(
    internal val inputArray1: AugmentedArray<InputNDArrayType>,
    internal val inputArray2: AugmentedArray<InputNDArrayType>,
    override val params: DistanceLayerParameters,
    activationFunction: ActivationFunction? = null,
    dropout: Double = 0.0,
    id: Int = 0
) :
    ItemsPool.IDItem,
    MergeLayer<InputNDArrayType>(
        inputArrays = listOf(inputArray1, inputArray2),
        outputArray = AugmentedArray(1),
        params = params,
        activationFunction = activationFunction,
        dropout = dropout,
        id = id) {

  init { this.checkInputSize() }

  /**
   * The helper which execute the forward.
   */
  override val forwardHelper = DistanceForwardHelper(layer = this)

  /**
   * The helper which execute the backward.
   */
  override val backwardHelper = DistanceBackwardHelper(layer = this)

  /**
   * The helper which calculates the relevance.
   */
  override val relevanceHelper = DistanceRelevanceHelper(layer = this)

  /**
   * @return the [DistanceLayerParameters] used to store errors
   */
  override fun parametersErrorsFactory() = DistanceLayerParameters(
      inputSize = this.params.inputSize)
}