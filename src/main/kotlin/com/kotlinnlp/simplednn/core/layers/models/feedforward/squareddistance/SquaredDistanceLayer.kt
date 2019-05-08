/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.squareddistance

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.layers.Layer
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.helpers.RelevanceHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.utils.ItemsPool

/**
 * The Squared Layer Structure.
 *
 * @property inputArray the input array of the layer
 * @property params the parameters which connect the input to the output
 * @property id an identification number useful to track a specific [SquaredDistanceLayer]
 */
class SquaredDistanceLayer(
    inputArray: AugmentedArray<DenseNDArray>,
    inputType: LayerType.Input,
    params: LayerParameters<*>,
    activationFunction: ActivationFunction? = null,
    dropout: Double = 0.0,
    override val id: Int = 0
) : ItemsPool.IDItem,
    Layer<DenseNDArray>(
        inputArray = inputArray,
        inputType = inputType,
        outputArray = AugmentedArray(1),
        params = params,
        activationFunction = activationFunction,
        dropout = dropout
    ) {
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
