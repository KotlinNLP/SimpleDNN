/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.convolution

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.helpers.BackwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory


/**
 * The helper which executes the backward on a [layer].
 *
 * @property layer the [ConvolutionLayer] in which the backward is executed
 */
class ConvolutionBackwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
    override val layer: ConvolutionLayer<InputNDArrayType>
) : BackwardHelper<InputNDArrayType>(layer) {

  /**
   * Initialize input errors
   */
  private fun initializeInputErrors() {
    this.layer.inputArrays.forEach{
      it.assignErrors(DenseNDArrayFactory.zeros(this.layer.inputSize))
    }
  }

  /**
   * Iniztialize parameters errors to zero
   * @param paramsErrors the parameters to initialize
   */
  private fun initializeParametersErrors(paramsErrors: ConvolutionLayerParameters) {
    paramsErrors.paramsList.forEach{
      it.errors.values.assignValues(n = 0.0)
    }
  }

  /**
   * Propagate errors to kernels (the layer parameters) from the output matrix at channel [outChannelIndex]
   *
   * @param outRowIndex the row index of the output matrix
   * @param outColIndex the column index of the output matrix
   * @param params the parameters to update with errors
   * @param outChannelIndex the index of the output channel
   */
  private fun calculateKernelsErrors(outChannelIndex: Int, outRowIndex: Int, outColIndex: Int,
                                     params: ConvolutionLayerParameters) {
    val startingIndex: Int = outChannelIndex * this.layer.params.inputChannels

    for (k in startingIndex until startingIndex + params.inputChannels) {

      for (i in (outRowIndex * this.layer.xStride) until (outRowIndex * this.layer.xStride)
          + params.kernelSize.dim1)
        for (j in (outColIndex * this.layer.yStride) until (outColIndex * this.layer.yStride)
            + params.kernelSize.dim2){
          params.paramsList[k].errors.values[i - (outRowIndex * this.layer.xStride),
              j - (outColIndex * this.layer.yStride)] =
              params.paramsList[k].errors.values[i - (outRowIndex * this.layer.xStride),
                  j - (outColIndex * this.layer.yStride)].toDouble() +
              (this.layer.inputArrays[k % this.layer.params.inputChannels].values[i, j].toDouble() *
                  this.layer.outputArrays[outChannelIndex].errors[outRowIndex, outColIndex])

        }
    }

  }

  /**
   * Perform inverse convolution, in order to propagate errors to input layers
   *
   * @param outRowIndex the row index of the output matrix
   * @param outColIndex the column index of the output matrix
   * @param outChannelIndex the index of the output channel
   */
  private fun inverseConvolution(outChannelIndex: Int, outRowIndex: Int, outColIndex: Int,
                                 params: ConvolutionLayerParameters) {

    val startingIndex: Int = outChannelIndex * this.layer.params.inputChannels

    for (k in startingIndex until startingIndex + params.inputChannels) {

      for (i in (outRowIndex * this.layer.xStride) until (outRowIndex * this.layer.xStride)
          + params.kernelSize.dim1)
        for (j in (outColIndex * this.layer.yStride) until (outColIndex * this.layer.yStride)
            + params.kernelSize.dim2){

          this.layer.inputArrays[k % this.layer.params.inputChannels].errors[i, j] +=
              (this.layer.params.paramsList[k].values[i - (outRowIndex * this.layer.xStride),
                  j - (outColIndex * this.layer.yStride)] *
                  this.layer.outputArrays[outChannelIndex].errors[outRowIndex, outColIndex])
        }

    }

  }

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array at index [outChannelIndex]
   *
   * @param outChannelIndex the index of the output channel
   * @param outputArray the output array at the index [outChannelIndex]
   * @param propagateToInput whether to propagate the errors to the input array
   */
  private fun backwardChannel(outChannelIndex: Int, outputArray: AugmentedArray<DenseNDArray>,
                              params: ConvolutionLayerParameters, propagateToInput: Boolean){


    val nWeights: Int = this.layer.params.outputChannels * this.layer.params.inputChannels

    for (r in 0 until outputArray.values.rows)
      for (c in 0 until outputArray.values.columns) {
        // propagate output errors to bias
        params.paramsList[nWeights + outChannelIndex].errors.values[0] = outputArray.errors[r, c] +
            params.paramsList[nWeights + outChannelIndex].errors.values[0].toDouble()

        calculateKernelsErrors(outChannelIndex, r, c, params)

        if (propagateToInput)
          inverseConvolution(outChannelIndex, r, c, params)
      }

  }

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output arrays.
   *
   * @param propagateToInput whether to propagate the errors to the input array
   */
  override fun execBackward(propagateToInput: Boolean) {
    this.layer.applyOutputActivationDerivs()
    this.initializeParametersErrors(this.layer.params)
    if (propagateToInput)
      initializeInputErrors()

    for ((index, outputArray) in this.layer.outputArrays.withIndex()) {
      require(outputArray.values.shape == outputArray.errors.shape)
      backwardChannel(index, outputArray, this.layer.params, propagateToInput)
    }
  }

}
