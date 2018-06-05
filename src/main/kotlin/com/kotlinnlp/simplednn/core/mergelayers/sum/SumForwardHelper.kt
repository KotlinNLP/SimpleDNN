/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.mergelayers.sum

import com.kotlinnlp.simplednn.core.layers.ForwardHelper
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the forward on an concat [layer].
 *
 * @property layer the [SumLayerStructure] in which the forward is executed
 */
class SumForwardHelper(override val layer: SumLayerStructure) : ForwardHelper<DenseNDArray>(layer) {

  /**
   * Forward the input to the output concatenating the input arrays.
   */
  override fun forward() = this.layer.inputArrays.forEachIndexed { i, x ->
    if (i == 0)
      this.layer.outputArray.assignValues(x.values)
    else
      this.layer.outputArray.values.assignSum(x.values)
  }

  /**
   * Forward the input to the output saving the contributions.
   * Not available for the Sum layer.
   *
   * @param layerContributions the structure in which to save the contributions during the calculations
   */
  override fun forward(layerContributions: LayerParameters<*>) {
    throw NotImplementedError("Forward with contributions not available for the Sum layer.")
  }
}
