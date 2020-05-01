/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.norm

import com.kotlinnlp.simplednn.core.layers.helpers.BackwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The helper which executes the backward on the [NormLayer].
 */
internal class NormBackwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: NormLayer<InputNDArrayType>
) : BackwardHelper<InputNDArrayType>(layer) {

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param propagateToInput whether to propagate the errors to the input array
   */
  override fun execBackward(propagateToInput: Boolean) {

    val gy: DenseNDArray = this.layer.outputArray.errors

    this.layer.params.b.errors.values.assignValues(gy)

    val gg: DenseNDArray = DenseNDArrayFactory.zeros(this.layer.inputArray.values.shape)
    gg.assignValues(this.layer.inputArray.values)
    gg.assignSub(this.layer.mean).assignDiv(this.layer.stdDev)

    this.layer.params.g.errors.values.assignValues(gg.assignProd(gy))

    if (propagateToInput)
      this.layer.inputArray.assignErrorsByProd(gy, this.layer.params.g.values.div(this.layer.stdDev))
  }
}
