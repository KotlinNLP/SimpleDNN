/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.maxpooling

import com.kotlinnlp.simplednn.core.layers.helpers.BackwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The helper which executes the backward on a [layer].
 *
 * @property layer the [MaxPoolingLayer] in which the backward is executed
 */
class MaxPoolingBackwardHelper <InputNDArrayType : NDArray<InputNDArrayType>>(
    override val layer: MaxPoolingLayer<InputNDArrayType>
) : BackwardHelper<InputNDArrayType>(layer)  {

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param propagateToInput whether to propagate the errors to the input array
   *
   */
  override fun execBackward(propagateToInput: Boolean) {

    require(this.layer.outputArray.values.shape ==this.layer.outputArray.errors.shape)

    this.layer.applyOutputActivationDeriv()

    if (propagateToInput) {
      this.layer.inputArray.assignErrors(DenseNDArrayFactory.zeros(this.layer.inputSize))

      for (r in 0 until layer.outputArray.values.rows)
        for (c in 0 until layer.outputArray.values.columns)
          this.layer.inputArray.errors[this.layer.argMaxi[r][c], this.layer.argMaxj[r][c]] =
              this.layer.outputArray.errors[r, c]
    }
  }

}