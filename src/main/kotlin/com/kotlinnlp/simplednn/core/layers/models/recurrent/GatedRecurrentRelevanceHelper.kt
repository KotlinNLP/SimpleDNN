/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.recurrent

import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which calculates the relevance of the input of a gated recurrent [layer] respect of its output.
 *
 * @property layer the [RecurrentLayer] in which to calculate the input relevance
 */
internal abstract class GatedRecurrentRelevanceHelper(
  layer: RecurrentLayer<DenseNDArray>
) : RecurrentRelevanceHelper(layer) {

  /**
   * Propagate the relevance from the output to the gates.
   *
   * @param contributions the contributions saved during the last forward
   */
  abstract fun propagateRelevanceToGates(contributions: LayerParameters)
}
