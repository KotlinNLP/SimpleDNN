/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.deltarnn

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.BackwardHelper
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the backward on a [layer].
 *
 * @property layer the [DeltaRNNLayerStructure] in which the backward is executed
 */
class DeltaRNNBackwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: DeltaRNNLayerStructure<InputNDArrayType>
) : BackwardHelper<InputNDArrayType> {
  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param paramsErrors the errors of the parameters which will be filled
   * @param propagateToInput whether to propagate the errors to the input array
   * @param mePropK the k factor of the 'meProp' algorithm to propagate from top k (in percentage) output nodes
   *                (ignored if null)
   */
  override fun backward(paramsErrors: LayerParameters<*>, propagateToInput: Boolean, mePropK: Double?) {

    // TODO: implement 'meProp' algorithm

    val prevStateOutput = this.layer.layerContextWindow.getPrevStateLayer()?.outputArray
    val nextStateLayer = this.layer.layerContextWindow.getNextStateLayer()

    if (nextStateLayer != null) {
      this.addOutputRecurrentGradients(nextStateLayer as DeltaRNNLayerStructure<*>)
    }

    this.layer.applyOutputActivationDeriv() // must be applied AFTER having added the recurrent gradients

    this.assignArraysGradients(prevStateOutput)
    this.assignParamsGradients(
      paramsErrors = paramsErrors as DeltaRNNLayerParameters,
      prevStateOutput = prevStateOutput)

    if (propagateToInput) {
      this.assignInputGradients(prevStateOutput)
    }
  }

  /**
   * Assign the errors to the candidate and the partition array.
   *
   * @param prevStateOutput the output array in the previous state
   */
  private fun assignArraysGradients(prevStateOutput: AugmentedArray<DenseNDArray>?) {

    val gy: DenseNDArray = this.layer.outputArray.errors

    val p: DenseNDArray = this.layer.partition.values
    val c: DenseNDArray = this.layer.candidate.values
    val pDeriv: DenseNDArray = this.layer.partition.calculateActivationDeriv()
    val cDeriv: DenseNDArray = this.layer.candidate.calculateActivationDeriv()

    val gpTmp: DenseNDArray = if (prevStateOutput != null) c.sub(prevStateOutput.values) else c

    this.layer.partition.assignErrorsByProd(gy, pDeriv).assignProd(gpTmp)
    this.layer.candidate.assignErrorsByProd(gy, cDeriv).assignProd(p)
  }

  /**
   * Add the errors coming from the next state to the output array.
   *
   * @param nextStateLayer the layer structure in the next state
   */
  private fun addOutputRecurrentGradients(nextStateLayer: DeltaRNNLayerStructure<*>) {

    val gy: DenseNDArray = this.layer.outputArray.errors
    val gyRec: DenseNDArray = this.getLayerRecurrentContribution(nextStateLayer)

    gy.assignSum(gyRec)
  }

  /**
   * @param nextStateLayer the layer structure in the next state
   *
   * @return the error of the output in the next state in respect of the current state
   */
  private fun getLayerRecurrentContribution(nextStateLayer: DeltaRNNLayerStructure<*>): DenseNDArray {

    this.layer.params as DeltaRNNLayerParameters

    val gyNext: DenseNDArray = nextStateLayer.outputArray.errors
    val gcNext: DenseNDArray = nextStateLayer.candidate.errors
    val pNext: DenseNDArray = nextStateLayer.partition.values
    val wxNext: DenseNDArray = nextStateLayer.wx.values

    val wRec: DenseNDArray = this.layer.params.recurrentUnit.weights.values as DenseNDArray
    val alpha: DenseNDArray = this.layer.params.alpha.values
    val beta2: DenseNDArray = this.layer.params.beta2.values

    val gRec1: DenseNDArray = pNext.reverseSub(1.0).assignProd(gyNext)
    val gRec2: DenseNDArray = alpha.prod(wxNext).assignSum(beta2).assignProd(gcNext).T.dot(wRec)

    return gRec1.assignSum(gRec2)
  }

  /**
   * Assign the errors to the parameters of the layer.
   *
   * @param paramsErrors the errors of the parameters which will be filled
   * @param prevStateOutput the output array in the previous state
   */
  private fun assignParamsGradients(paramsErrors: DeltaRNNLayerParameters,
                                    prevStateOutput: AugmentedArray<DenseNDArray>?) {

    this.layer.params as DeltaRNNLayerParameters

    val x: InputNDArrayType = this.layer.inputArray.values

    val gp: DenseNDArray = this.layer.partition.errors
    val gc: DenseNDArray = this.layer.candidate.errors

    val wx: DenseNDArray = this.layer.wx.values
    val wyRec: DenseNDArray = this.layer.wyRec.values
    val alpha: DenseNDArray = this.layer.params.alpha.values
    val beta1: DenseNDArray = this.layer.params.beta1.values
    val beta2: DenseNDArray = this.layer.params.beta2.values

    val gw: NDArray<*> = paramsErrors.feedforwardUnit.weights.values
    val gbc: DenseNDArray = paramsErrors.feedforwardUnit.biases.values
    val gbp: DenseNDArray = paramsErrors.recurrentUnit.biases.values
    val gAlpha: DenseNDArray = paramsErrors.alpha.values
    val gBeta1: DenseNDArray = paramsErrors.beta1.values
    val gBeta2: DenseNDArray = paramsErrors.beta2.values

    gbc.assignValues(gc)
    gbp.assignValues(gp)

    gBeta1.assignProd(gc, wx)
    gBeta2.assignProd(gc, wyRec)
    gAlpha.assignProd(gBeta1, wyRec)

    val gwTmp: DenseNDArray = if (prevStateOutput != null) alpha.prod(wyRec).assignSum(beta1) else beta1.copy()
    gwTmp.assignProd(gc).assignSum(gp)
    gw.assignDot(gwTmp, x.T)

    if (prevStateOutput != null) {
      val gwRec: DenseNDArray = paramsErrors.recurrentUnit.weights.values as DenseNDArray
      val yPrev: DenseNDArray = prevStateOutput.values

      val gwRecTmp: DenseNDArray = alpha.prod(wx).assignSum(beta2).assignProd(gc)
      gwRec.assignDot(gwRecTmp, yPrev.T)
    }
  }

  /**
   * Assign the errors to the input array.
   *
   * @param prevStateOutput the output array in the previous state
   */
  private fun assignInputGradients(prevStateOutput: AugmentedArray<DenseNDArray>?) {

    this.layer.params as DeltaRNNLayerParameters

    val gp: DenseNDArray = this.layer.partition.errors
    val gc: DenseNDArray = this.layer.candidate.errors

    val w: NDArray<*> = this.layer.params.feedforwardUnit.weights.values
    val wyRec: DenseNDArray = this.layer.wyRec.values
    val alpha: DenseNDArray = this.layer.params.alpha.values
    val beta1: DenseNDArray = this.layer.params.beta1.values

    val gxTmp: DenseNDArray = if (prevStateOutput != null) alpha.prod(wyRec).assignSum(beta1) else beta1
    gxTmp.assignProd(gc).assignSum(gp)

    this.layer.inputArray.assignErrorsByDotT(gxTmp.T, w)
  }
}
