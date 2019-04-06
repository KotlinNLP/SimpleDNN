/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.reshape

import com.kotlinnlp.simplednn.core.layers.helpers.BackwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

class ReshapeBackwardHelper <InputNDArrayType : NDArray<InputNDArrayType>>(
    override val layer: ReshapeLayer<InputNDArrayType>
) : BackwardHelper<InputNDArrayType>(layer) {

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param propagateToInput whether to propagate the errors to the input array
   *
   */
  override fun execBackward(propagateToInput: Boolean) {
    require(this.layer.outputArray.values.shape ==this.layer.outputArray.errors.shape)
    if (propagateToInput) {
      var i = 0
      val outputErrors = DoubleArray(this.layer.outputArray.size)
      for (r in 0 until layer.outputArray.values.rows)
        for (c in 0 until layer.outputArray.values.columns){

          outputErrors[i] = layer.outputArray.errors[r, c]
          i++

        }
      i = 0
      this.layer.inputArray.assignErrors(DenseNDArrayFactory.zeros(this.layer.inputSize))
      for (r in 0 until layer.inputArray.values.rows)
        for (c in 0 until layer.inputArray.values.columns){

          layer.inputArray.errors[r, c] = outputErrors[i]
          i++

        }

    }
  }

}