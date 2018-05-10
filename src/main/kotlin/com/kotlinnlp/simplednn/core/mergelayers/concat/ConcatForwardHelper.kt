/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.mergelayers.concat

import com.kotlinnlp.simplednn.core.layers.ForwardHelper
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the forward on an concat [layer].
 *
 * @property layer the [ConcatLayerStructure] in which the forward is executed
 */
class ConcatForwardHelper(override val layer: ConcatLayerStructure) : ForwardHelper<DenseNDArray>(layer) {

  /**
   * Forward the input to the output concatenating the input arrays.
   */
  override fun forward() {

    val x1: DenseNDArray = this.layer.inputArray.values
    val x2: DenseNDArray = this.layer.inputArray2.values

    this.layer.outputArray.assignValues(x1.concatV(x2))
  }

  /**
   * Forward the input to the output saving the contributions.
   *
   * @param layerContributions the structure in which to save the contributions during the calculations
   */
  override fun forward(layerContributions: LayerParameters<*>) {
    layerContributions as ConcatLayerParameters

    TODO("not implemented")
  }
}