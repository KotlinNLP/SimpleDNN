/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.normalization

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

    this.layer.applyOutputActivationDerivs()

    this.layer.outputArrays.forEachIndexed { index, outputArray ->

      val gy: DenseNDArray = outputArray.errors

      this.layer.params.b.errors.values.assignSum(gy)

      val sub: DenseNDArray = DenseNDArrayFactory.zeros(this.layer.inputArrays[0].values.shape)
      sub.assignValues(this.layer.inputArrays[index].values)
      sub.assignSub(this.layer.meanArray).assignDiv(this.layer.devStdArray.assignSum(0.00000000001))

      this.layer.params.g.errors.values.assignSum(sub.assignProd(gy))

      if (propagateToInput) {
        this.layer.inputArrays[index].assignErrors(
          gy.assignProd(this.layer.params.g.values.div(this.layer.devStdArray)))
      }
    }
  }
}
