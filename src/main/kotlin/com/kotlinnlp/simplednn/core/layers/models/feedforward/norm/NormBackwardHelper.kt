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
    val devStdDev: InputNDArrayType = this.layer.devStdDev

    this.layer.params.b.errors.values.assignValues(gy)
    this.layer.params.g.errors.values.assignValues(gy).assignProd(devStdDev)

    if (propagateToInput) {

      val n: Double = this.layer.inputArray.size.toDouble()
      val g: DenseNDArray = this.layer.params.g.values
      val v: Double = this.layer.v
      val dev: InputNDArrayType = this.layer.dev
      val stdDev: Double = this.layer.stdDev

      val gyG: DenseNDArray = gy.prod(g)
      val gxDev: DenseNDArray = gyG.div(stdDev)
      val gxDevXm: Double = -gxDev.sum() / n

      val gxV: Double = -gyG.assignProd(dev).assignDiv(2.0 * (v + NormLayer.EPS) * stdDev).sum() / n
      val gxVx: InputNDArrayType = dev.prod(2.0)
      val gxVxm: Double = -gxVx.sum() / n

      val gx: DenseNDArray = gxDev.assignSum(gxDevXm).assignSum(gxVx.assignSum(gxVxm).assignProd(gxV))

      this.layer.inputArray.assignErrors(gx)
    }
  }
}
