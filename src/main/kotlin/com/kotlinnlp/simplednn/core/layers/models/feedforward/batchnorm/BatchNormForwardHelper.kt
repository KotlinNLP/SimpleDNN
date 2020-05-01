/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.batchnorm

import com.kotlinnlp.simplednn.core.layers.helpers.ForwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import kotlin.math.sqrt

/**
 * The helper which executes the forward on the [BatchNormLayer].
 *
 * @param layer the layer with which this helper works
 */
internal class BatchNormForwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: BatchNormLayer<InputNDArrayType>
) : ForwardHelper<InputNDArrayType>(layer) {

  companion object {

    /**
     * Avoid underflow errors.
     */
    private const val EPS = 1.0e-12
  }

  /**
   * Forward the input to the output combining it with the parameters.
   */
  override fun forward() {

    // Important: the mean must be calculated before the std dev!
    calculateMean()
    calculateStdDev()

    val gStdDev: DenseNDArray = this.layer.params.g.values.div(this.layer.stdDev)

    this.layer.inputArrays.zip(this.layer.outputArrays).forEach { (input, output) ->

      output.values
        .assignValues(input.values)
        .assignSub(this.layer.mean)
        .assignProd(gStdDev)
        .assignSum(this.layer.params.b.values)

      output.activate()
    }
  }

  /**
   * Calculate the standard deviation of the input arrays and assign it to the support array `this.layer.stdDev`.
   */
  private fun calculateStdDev() {

    this.layer.stdDev.zeros()

    this.layer.inputArrays.forEach { input ->

      val diff: DenseNDArray = this.layer.mean.sub(input.values).assignPow(2.0)

      this.layer.stdDev.assignSum(diff)
    }

    (0 until this.layer.stdDev.length).forEach { i ->
      this.layer.stdDev[i] = sqrt(this.layer.stdDev[i] / this.layer.inputArrays.size + EPS)
    }
  }

  /**
   * Calculate the mean of the input arrays and assign it to the support array `this.layer.mean`.
   */
  private fun calculateMean() {

    this.layer.mean.zeros()

    this.layer.inputArrays.forEach {
      this.layer.mean.assignSum(it.values)
    }

    this.layer.mean.assignDiv(this.layer.inputArrays.size.toDouble())
  }
}
