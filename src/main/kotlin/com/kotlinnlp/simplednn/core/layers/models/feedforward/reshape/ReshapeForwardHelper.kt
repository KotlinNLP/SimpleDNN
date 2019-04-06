/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.reshape

import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.helpers.ForwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * The helper which executes the forward on a [layer].
 *
 * @property layer the [ReshapeLayer] in which the forward is executed
 */
class ReshapeForwardHelper <InputNDArrayType : NDArray<InputNDArrayType>>(
    override val layer: ReshapeLayer<InputNDArrayType>
) : ForwardHelper<InputNDArrayType>(layer) {


  /**
   * Forward the input to the output
   *
   */
  override fun forward() {
    var i = 0
    val outputArray = DoubleArray(this.layer.outputArray.size)
    for (r in 0 until layer.inputArray.values.rows)
      for (c in 0 until layer.inputArray.values.columns){

        outputArray[i] = layer.inputArray.values[r, c].toDouble()
        i++

      }
    i = 0
    for (r in 0 until layer.outputArray.values.rows)
      for (c in 0 until layer.outputArray.values.columns){
        this.layer.outputArray.values[r, c] = outputArray[i]
        i++
      }
  }

  /**
   * Forward the input to the output, saving the contributions.
   *
   */
  override fun forward(layerContributions: LayerParameters<*>) {
    TODO("not implemented")
  }
}