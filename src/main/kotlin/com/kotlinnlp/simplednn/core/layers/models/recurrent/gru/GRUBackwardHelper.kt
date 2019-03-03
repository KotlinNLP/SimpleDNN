/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.recurrent.gru

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.helpers.BackwardHelper
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the backward on a [layer].
 *
 * @property layer the [GRULayer] in which the backward is executed
 */
class GRUBackwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: GRULayer<InputNDArrayType>
) : BackwardHelper<InputNDArrayType> {

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param paramsErrors the errors of the parameters which will be filled
   * @param propagateToInput whether to propagate the errors to the input array
   */
  override fun backward(paramsErrors: LayerParameters<*>, propagateToInput: Boolean) {

    val prevStateOutput = this.layer.layerContextWindow.getPrevState()?.outputArray
    val nextStateLayer = this.layer.layerContextWindow.getNextState()

    this.addOutputRecurrentGradients(nextStateLayer as? GRULayer<*>)

    this.assignGatesGradients(prevStateOutput)

    this.assignParamsGradients(
      paramsErrors = paramsErrors as GRULayerParameters,
      prevStateOutput = prevStateOutput)

    if (propagateToInput) {
      this.assignLayerGradients()
    }
  }

  /**
   * @param prevStateOutput the outputArray in the previous state
   */
  private fun assignGatesGradients(prevStateOutput: AugmentedArray<DenseNDArray>?) {

    this.layer.params as GRULayerParameters

    val gy: DenseNDArray = this.layer.outputArray.errors

    val p: DenseNDArray = this.layer.partitionGate.values
    val c: DenseNDArray = this.layer.candidate.values

    val rDeriv: DenseNDArray = this.layer.resetGate.calculateActivationDeriv()
    val pDeriv: DenseNDArray = this.layer.partitionGate.calculateActivationDeriv()

    this.layer.candidate.assignErrorsByProd(p, gy) // gc must be calculated before gr and gp

    if (this.layer.candidate.hasActivation) {
      val cDeriv: DenseNDArray = this.layer.candidate.calculateActivationDeriv()
      this.layer.candidate.errors.assignProd(cDeriv)
    }

    if (prevStateOutput == null) {
      this.layer.resetGate.assignZeroErrors()
      this.layer.partitionGate.assignErrorsByProd(c, pDeriv).assignProd(gy)

    } else { // recurrent contribution
      val gc: DenseNDArray = this.layer.candidate.errors
      val yPrev: DenseNDArray = prevStateOutput.values
      val wcr: DenseNDArray = this.layer.params.candidate.recurrentWeights.values as DenseNDArray

      this.layer.resetGate.assignErrorsByDotT(gc.t, wcr).assignProd(rDeriv).assignProd(yPrev)
      this.layer.partitionGate.assignErrorsByProd(c.sub(yPrev), pDeriv).assignProd(gy)
    }
  }

  /**
   * @param paramsErrors the errors of the parameters which will be filled
   * @param prevStateOutput the outputArray in the previous state
   */
  private fun assignParamsGradients(paramsErrors: GRULayerParameters, prevStateOutput: AugmentedArray<DenseNDArray>?) {

    val x: InputNDArrayType = this.layer.inputArray.values
    val yPrev: DenseNDArray? = prevStateOutput?.values

    this.layer.resetGate.assignParamsGradients(paramsErrors = paramsErrors.resetGate, x = x, yPrev = yPrev)
    this.layer.partitionGate.assignParamsGradients(paramsErrors = paramsErrors.partitionGate, x = x, yPrev = yPrev)
    this.layer.candidate.assignParamsGradients(paramsErrors = paramsErrors.candidate, x = x)

    if (yPrev != null) { // add recurrent contribution to the recurrent weights of the candidate
      val r: DenseNDArray = this.layer.resetGate.values
      val gwcr: DenseNDArray = paramsErrors.candidate.recurrentWeights.values as DenseNDArray
      val gc: DenseNDArray = this.layer.candidate.errors
      gwcr.assignDot(gc, r.prod(yPrev).t)
    }
  }

  /**
   *
   */
  private fun assignLayerGradients() {

    this.layer.params as GRULayerParameters

    val wp: DenseNDArray = this.layer.params.partitionGate.weights.values as DenseNDArray
    val wc: DenseNDArray = this.layer.params.candidate.weights.values as DenseNDArray
    val wr: DenseNDArray = this.layer.params.resetGate.weights.values as DenseNDArray

    val gp: DenseNDArray = this.layer.partitionGate.errors
    val gc: DenseNDArray = this.layer.candidate.errors
    val gr: DenseNDArray = this.layer.resetGate.errors

    this.layer.inputArray
      .assignErrorsByDotT(gp.t, wp)
      .assignSum(gc.t.dot(wc))
      .assignSum(gr.t.dot(wr))
  }

  /**
   * @param nextStateLayer the layer structure in the next state
   */
  private fun addOutputRecurrentGradients(nextStateLayer: GRULayer<*>?) {

    if (nextStateLayer != null) {
      val gy: DenseNDArray = this.layer.outputArray.errors
      val gyRec: DenseNDArray = this.getLayerRecurrentContribution(nextStateLayer)

      gy.assignSum(gyRec)
    }
  }

  /**
   * @param nextStateLayer the layer structure in the next state
   */
  private fun getLayerRecurrentContribution(nextStateLayer: GRULayer<*>): DenseNDArray {

    this.layer.params as GRULayerParameters

    val resetGate = nextStateLayer.resetGate
    val partitionGate = nextStateLayer.partitionGate
    val candidate = nextStateLayer.candidate

    val gy: DenseNDArray = nextStateLayer.outputArray.errors

    val r: DenseNDArray = resetGate.values
    val p: DenseNDArray = partitionGate.values

    val gr: DenseNDArray = resetGate.errors
    val gp: DenseNDArray = partitionGate.errors
    val gc: DenseNDArray = candidate.errors

    val wrr: DenseNDArray = this.layer.params.resetGate.recurrentWeights.values as DenseNDArray
    val wpr: DenseNDArray = this.layer.params.partitionGate.recurrentWeights.values as DenseNDArray
    val wcr: DenseNDArray = this.layer.params.candidate.recurrentWeights.values as DenseNDArray

    val gRec1: DenseNDArray = gr.t.dot(wrr)
    val gRec2: DenseNDArray = gp.t.dot(wpr)
    val gRec3: DenseNDArray = gc.t.dot(wcr).prod(r)
    val gRec4: DenseNDArray = p.reverseSub(1.0).assignProd(gy).t

    return gRec1.assignSum(gRec2).assignSum(gRec3).assignSum(gRec4)
  }
}
