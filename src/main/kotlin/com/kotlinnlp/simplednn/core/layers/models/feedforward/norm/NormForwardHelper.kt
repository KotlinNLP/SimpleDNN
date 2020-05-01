/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.norm

import com.kotlinnlp.simplednn.core.layers.helpers.ForwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import kotlin.math.sqrt

/**
 * The helper which executes the forward on the [NormLayer].
 *
 * @param layer the layer with which this helper works
 */
internal class NormForwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: NormLayer<InputNDArrayType>
) : ForwardHelper<InputNDArrayType>(layer) {

  companion object {

    /**
     * Avoid underflow errors.
     */
    private const val EPS = 1.0e-5
  }

  /**
   * Forward the input to the output combining it with the parameters.
   *
   *  y = (x - E\[x\]) / sqrt(VAR\[x\] + [EPS]) * g + b
   */
  override fun forward() {

    val x: InputNDArrayType = this.layer.inputArray.values
    val y: DenseNDArray = this.layer.outputArray.values
    val g: DenseNDArray = this.layer.params.g.values
    val b: DenseNDArray = this.layer.params.b.values

    val mean: Double = x.avg()
    val dev: InputNDArrayType = x.sub(mean)
    val stdDev: Double = sqrt(dev.pow(2.0).avg() + EPS)

    y.assignValues(dev).assignDiv(stdDev).assignProd(g).assignSum(b)

    this.layer.mean = mean
    this.layer.stdDev = stdDev
  }
}
