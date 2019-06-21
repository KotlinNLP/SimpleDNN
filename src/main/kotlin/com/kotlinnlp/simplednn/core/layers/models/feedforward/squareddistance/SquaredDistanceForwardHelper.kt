/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.squareddistance

import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.helpers.ForwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * The helper which executes the forward on a [SquaredDistanceLayer].
 *
 * @property layer the layer in which the forward is executed
 */
class SquaredDistanceForwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: SquaredDistanceLayer<InputNDArrayType>
) : ForwardHelper<InputNDArrayType>(layer){

  /**
   * Forward the input to the output calculating a score value d >= 0.
   * The output is `BhT dot Bh`, where `B` is the parameter matrix, and `h` is the input.
   */
  override fun forward() {

    this.layer.params as SquaredDistanceLayerParameters

    this.layer.bhOut.assignValues(layer.params.wB.values.dot(layer.inputArray.values))

    this.layer.bhOut.values.let { bhOut ->
      this.layer.outputArray.assignValues(bhOut.t.dot(bhOut))
    }
  }
}