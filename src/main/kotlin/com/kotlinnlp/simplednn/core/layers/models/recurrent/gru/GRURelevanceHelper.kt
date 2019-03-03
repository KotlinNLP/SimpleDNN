/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.recurrent.gru

import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.models.recurrent.GatedRecurrentRelevanceHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * The helper which calculates the relevance of the input of a [layer] respect of its output.
 *
 * @property layer the [GRULayer] in which to calculate the input relevance
 */
class GRURelevanceHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  layer: GRULayer<InputNDArrayType>
) : GatedRecurrentRelevanceHelper<InputNDArrayType>(layer) {

  /**
   * Propagate the relevance from the output to the gated units of the layer.
   *
   * @param layerContributions the structure in which to save the contributions during the calculations
   */
  override fun propagateRelevanceToGates(layerContributions: LayerParameters<*>) {
    TODO("not implemented")
  }

  /**
   * @param layerContributions the contributions saved during the last forward
   *
   * @return the relevance of the input respect of the output
   */
  override fun getInputRelevance(layerContributions: LayerParameters<*>): NDArray<*> {
    TODO("not implemented")
  }

  /**
   * Calculate the relevance of the output in the previous state respect of the current one and assign it to the output
   * array of the previous state.
   * WARNING: it's needed that a previous state exists.
   *
   * @param layerContributions the contributions saved during the last forward
   */
  override fun setRecurrentRelevance(layerContributions: LayerParameters<*>) {
    TODO("not implemented")
  }
}
