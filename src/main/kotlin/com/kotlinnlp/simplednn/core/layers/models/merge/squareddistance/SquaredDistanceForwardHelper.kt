/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.merge.squareddistance

import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.helpers.ForwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the forward on a [SquaredDistanceLayer].
 *
 * @property layer the layer in which the forward is executed
 */
class SquaredDistanceForwardHelper(override val layer: SquaredDistanceLayer) : ForwardHelper<DenseNDArray>(layer){

  /**
   * Forward the input to the output calculating a score value d >= 0.
   * output is BhT Bh where B is the parameter matrix, and h is the input
   */
  override fun forward() {


  }

  /**
   * Forward the input to the output saving the contributions.
   * Not available for the Squared Distance layer.
   *
   * @param layerContributions the structure in which to save the contributions during the calculations
   */
  override fun forward(layerContributions: LayerParameters<*>) {
    throw NotImplementedError("Forward with contributions not available for the Distance layer.")
  }
}