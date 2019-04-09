/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.maxpooling

import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.helpers.ForwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * The helper which executes the forward on a [layer].
 *
 * @property layer the [MaxPoolingLayer] in which the forward is executed
 */
class MaxPoolingForwardHelper <InputNDArrayType : NDArray<InputNDArrayType>>(
    override val layer: MaxPoolingLayer<InputNDArrayType>
) : ForwardHelper<InputNDArrayType>(layer) {


  /**
   * Perform max pooling, selecting max in the input sub-matrix starting from (row, col)
   *
   * @param row The row of the output matrix
   * @param col the column of the output matrix
   */
  private fun maxPooling(row: Int, col: Int) {

    var max : Double = Double.MIN_VALUE

    for (i in (row * this.layer.poolSize.dim1) until (row * this.layer.poolSize.dim1)
        + this.layer.poolSize.dim1)
      for (j in (col * this.layer.poolSize.dim2)  until (col * this.layer.poolSize.dim2)
          + this.layer.poolSize.dim1)
        if (this.layer.inputArray.values[i,j].toDouble() > max){

          max = this.layer.inputArray.values[i,j].toDouble()
          this.layer.argMaxi[row][col] = i
          this.layer.argMaxj[row][col] = j

        }

    this.layer.outputArray.values[row, col] = max
  }

  /**
   * Forward the input to the output
   *
   */
  override fun forward() {

    for (r in 0 until layer.outputArray.values.rows)
      for (c in 0 until layer.outputArray.values.columns)
        maxPooling(r, c)

    layer.outputArray.activate()
  }

  /**
   * Forward the input to the output, saving the contributions.
   *
   */
  override fun forward(layerContributions: LayerParameters<*>) {
    TODO("not implemented")
  }

}