/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.recurrent

import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.helpers.RelevanceHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which calculates the relevance of the input of a recurrent [layer] respect of its output.
 *
 * @property layer the [RecurrentLayer] in which to calculate the input relevance
 */
abstract class RecurrentRelevanceHelper(override val layer: RecurrentLayer<DenseNDArray>) : RelevanceHelper(layer) {

  /**
   * Calculate the relevance of the output in the previous state respect of the current one and assign it to the output
   * array of the previous state.
   * WARNING: it's needed that a previous state exists.
   *
   * @param layerContributions the structure in which to save the contributions during the calculations
   */
  abstract fun setRecurrentRelevance(layerContributions: LayerParameters)
}
