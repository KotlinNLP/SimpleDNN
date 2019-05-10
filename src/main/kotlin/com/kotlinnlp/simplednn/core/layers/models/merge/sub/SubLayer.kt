/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.merge.sub

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.helpers.RelevanceHelper
import com.kotlinnlp.simplednn.core.layers.models.merge.MergeLayer
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.utils.ItemsPool

/**
 * The Sub Layer Structure.
 *
 * @property inputArray the first input array of the layer
 * @property inputArray2 the second input array of the layer
 * @property params the parameters which connect the input to the output
 * @property id an identification number useful to track a specific [SubLayer]
 */
class SubLayer<InputNDArrayType : NDArray<InputNDArrayType>>(
  internal val inputArray1: AugmentedArray<InputNDArrayType>,
  internal val inputArray2: AugmentedArray<InputNDArrayType>,
  inputType: LayerType.Input,
  outputArray: AugmentedArray<DenseNDArray>,
  override val params: SubLayerParameters,
  id: Int = 0
) :
  ItemsPool.IDItem,
  MergeLayer<InputNDArrayType>(
    inputArrays = listOf(inputArray1, inputArray2),
    inputType = inputType,
    outputArray = outputArray,
    params = params,
    activationFunction = null,
    dropout = 0.0,
    id = id
  ) {

  init { this.checkInputSize() }

  /**
   * The helper which execute the forward.
   */
  override val forwardHelper = SubForwardHelper(layer = this)

  /**
   * The helper which execute the backward.
   */
  override val backwardHelper = SubBackwardHelper(layer = this)

  /**
   * The helper which calculates the relevance
   */
  override val relevanceHelper: RelevanceHelper? = null
}