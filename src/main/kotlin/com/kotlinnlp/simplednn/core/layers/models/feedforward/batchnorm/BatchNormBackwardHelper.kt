/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.batchnorm

import com.kotlinnlp.simplednn.core.layers.helpers.BackwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The helper which executes the backward on the [BatchNormLayer].
 */
internal class BatchNormBackwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: BatchNormLayer<InputNDArrayType>
) : BackwardHelper<InputNDArrayType>(layer) {

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param propagateToInput whether to propagate the errors to the input array
   */
  override fun execBackward(propagateToInput: Boolean) {

    val gIn: DenseNDArray? = if (propagateToInput) this.layer.params.g.values.div(this.layer.stdDev) else null

    this.layer.params.b.errors.values.zeros()
    this.layer.params.g.errors.values.zeros()

    this.layer.inputArrays.zip(this.layer.outputArrays).forEach { (input, output) ->

      val gy: DenseNDArray = output.errors

      this.layer.params.b.errors.values.assignSum(gy)

      val gg: DenseNDArray = DenseNDArrayFactory.zeros(input.values.shape)
      gg.assignValues(input.values)
      gg.assignSub(this.layer.mean).assignDiv(this.layer.stdDev)

      this.layer.params.g.errors.values.assignSum(gg.assignProd(gy))

      if (propagateToInput)
        input.assignErrorsByProd(gy, gIn!!)
    }
  }
}
