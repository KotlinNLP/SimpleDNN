/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.recurrent.cfn

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.helpers.BackwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the backward on a [layer].
 *
 * @property layer the [CFNLayer] in which the backward is executed
 */
class CFNBackwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: CFNLayer<InputNDArrayType>
) : BackwardHelper<InputNDArrayType>(layer) {

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param propagateToInput whether to propagate the errors to the input array
   */
  override fun execBackward(propagateToInput: Boolean) {

    val prevStateLayer = this.layer.layerContextWindow.getPrevState() as? CFNLayer
    val nextStateLayer = this.layer.layerContextWindow.getNextState() as? CFNLayer

    if (nextStateLayer != null) {
      this.addOutputRecurrentGradients(nextStateLayer)
    }

    this.assignGatesGradients(prevStateLayer)

    this.assignParamsGradients(prevStateOutput = prevStateLayer?.outputArray)

    if (propagateToInput) {
      this.assignLayerGradients()
    }
  }

  /**
   *
   * @param prevStateLayer the layer in the previous state
   */
  private fun assignGatesGradients(prevStateLayer: CFNLayer<*>?) {


    val gy: DenseNDArray = this.layer.outputArray.errors

    val inputGate = this.layer.inputGate
    val forgetGate = this.layer.forgetGate
    val candidate = this.layer.candidate

    val ingG: DenseNDArray = inputGate.values
    val c: DenseNDArray = candidate.values

    val inGDeriv: DenseNDArray = inputGate.calculateActivationDeriv()

    this.layer.inputGate.assignErrorsByProd(c, inGDeriv).assignProd(gy)
    this.layer.candidate.assignErrorsByProd(ingG, gy)

    if (this.layer.candidate.hasActivation) {
      val cDeriv: DenseNDArray = candidate.calculateActivationDeriv()
      this.layer.candidate.errors.assignProd(cDeriv)
    }

    if (prevStateLayer != null) {
      val aPrev: DenseNDArray = this.layer.activatedPrevOutput!!
      val forGDeriv: DenseNDArray = forgetGate.calculateActivationDeriv()

      this.layer.forgetGate.assignErrorsByProd(aPrev, forGDeriv).assignProd(gy)

    } else {
      this.layer.forgetGate.assignZeroErrors()
    }
  }

  /**
   * @param prevStateOutput the outputArray in the previous state
   */
  private fun assignParamsGradients(prevStateOutput: AugmentedArray<DenseNDArray>?) {

    val x: InputNDArrayType = this.layer.inputArray.values
    val yPrev: DenseNDArray? = prevStateOutput?.values

    this.layer.inputGate.assignParamsGradients(
      gw = this.layer.params.inputGate.weights.errors.values,
      gb = this.layer.params.inputGate.biases.errors.values,
      gwRec = this.layer.params.inputGate.recurrentWeights.errors.values,
      x = x,
      yPrev = yPrev)

    this.layer.forgetGate.assignParamsGradients(
      gw = this.layer.params.forgetGate.weights.errors.values,
      gb = this.layer.params.forgetGate.biases.errors.values,
      gwRec = this.layer.params.forgetGate.recurrentWeights.errors.values,
      x = x,
      yPrev = yPrev)

    val gc: DenseNDArray = this.layer.candidate.errors
    val gwc: NDArray<*> = this.layer.params.candidateWeights.errors.values
    gwc.assignDot(gc, x.t)
  }

  /**
   *
   */
  private fun assignLayerGradients() {

    val wInG: DenseNDArray = this.layer.params.inputGate.weights.values
    val wForG: DenseNDArray = this.layer.params.forgetGate.weights.values
    val wC: DenseNDArray = this.layer.params.candidateWeights.values

    val gInG: DenseNDArray = this.layer.inputGate.errors
    val gForG: DenseNDArray = this.layer.forgetGate.errors
    val gC: DenseNDArray = this.layer.candidate.errors

    this.layer.inputArray
      .assignErrorsByDotT(gForG.t, wForG)
      .assignSum(gC.t.dot(wC))
      .assignSum(gInG.t.dot(wInG))
  }

  /**
   *
   * @param nextStateLayer the layer structure in the next state
   */
  private fun addOutputRecurrentGradients(nextStateLayer: CFNLayer<*>) {

    val gy: DenseNDArray = this.layer.outputArray.errors
    val gyRec: DenseNDArray = this.getLayerRecurrentContribution(nextStateLayer)

    gy.assignSum(gyRec)
  }

  /**
   *
   * @param nextStateLayer the layer structure in the next state
   */
  private fun getLayerRecurrentContribution(nextStateLayer: CFNLayer<*>): DenseNDArray {

    val inputGate = nextStateLayer.inputGate
    val forgetGate = nextStateLayer.forgetGate

    val gyNext: DenseNDArray = nextStateLayer.outputArray.errors

    val yDeriv: DenseNDArray = if (nextStateLayer.activationFunction != null)
      nextStateLayer.activationFunction.dfOptimized(nextStateLayer.activatedPrevOutput!!)
    else
      nextStateLayer.activatedPrevOutput!!

    val forG: DenseNDArray = forgetGate.values

    val gInG: DenseNDArray = inputGate.errors
    val gForG: DenseNDArray = forgetGate.errors

    val wrInG: DenseNDArray = this.layer.params.inputGate.recurrentWeights.values
    val wrForG: DenseNDArray = this.layer.params.forgetGate.recurrentWeights.values

    val gRec1: DenseNDArray = forG.prod(yDeriv).assignProd(gyNext)
    val gRec2: DenseNDArray = gInG.t.dot(wrInG)
    val gRec3: DenseNDArray = gForG.t.dot(wrForG)

    return gRec1.assignSum(gRec2).assignSum(gRec3)
  }
}
