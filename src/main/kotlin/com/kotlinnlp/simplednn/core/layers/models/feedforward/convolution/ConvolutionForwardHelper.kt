/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.convolution

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.helpers.ForwardHelper
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the forward on a [layer].
 *
 * @property layer the [ConvolutionLayer] in which the forward is executed
 */
class ConvolutionForwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
    override val layer: ConvolutionLayer<InputNDArrayType>
) : ForwardHelper<InputNDArrayType>(layer) {

  /**
   * Perform convolution, giving the result for the position ([outRowIndex], [outColIndex])
   * in the output channel at the index [outChannelIndex]
   *
   * @param outRowIndex the row index of the output matrix
   * @param outColIndex the column index of the output matrix
   * @param outChannelIndex the index of the output channel
   */
  private fun convolution(outChannelIndex: Int, outRowIndex: Int, outColIndex: Int) : Double {
    var sum = 0.0
    val startingIndex: Int = outChannelIndex * this.layer.params.inputChannels

    for (k in startingIndex until startingIndex + this.layer.params.inputChannels) {
      for (i in (outRowIndex * this.layer.xStride) until  (outRowIndex * this.layer.xStride)
          + this.layer.params.kernelSize.dim1)
        for (j in (outColIndex * this.layer.yStride) until (outColIndex * this.layer.yStride)
            + this.layer.params.kernelSize.dim2)
          sum += (this.layer.inputArrays[k % this.layer.params.inputChannels].values[i, j].toDouble() *
              this.layer.params.paramsList[k].values[i -  (outRowIndex * this.layer.xStride),
                  j - (outColIndex * this.layer.yStride)])

    }
    return sum
  }

  /**
   * Forward the input to the output combining it with the parameters, in the output channel at the
   * index [outChannelIndex]
   *
   * @param outChannelIndex the index of the outputChannel
   * @param outputArray the  outputChannel at the index of [outChannelIndex]
   */
  private fun forwardChannel(outChannelIndex: Int, outputArray: AugmentedArray<DenseNDArray>){


    val nWeights: Int = this.layer.params.outputChannels * this.layer.params.inputChannels

    for (r in 0 until outputArray.values.rows)
      for (c in 0 until outputArray.values.columns)
        outputArray.values[r, c] = this.layer.params.paramsList[nWeights + outChannelIndex].values[0] +
            convolution(outChannelIndex, r, c)

    outputArray.activate()

  }

  /**
   * Forward the input to the output combining it with the parameters
   */
  override fun forward() {

    for ((index, outputArray) in this.layer.outputArrays.withIndex())
      forwardChannel(index, outputArray)

  }

  /**
   * Forward the input to the output combining it with the parameters, saving the contributions.
   *
   */
  override fun forward(layerContributions: LayerParameters<*>) {
    TODO("not implemented")
  }

}