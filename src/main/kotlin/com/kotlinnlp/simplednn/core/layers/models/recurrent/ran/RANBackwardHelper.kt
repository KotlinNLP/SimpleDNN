/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.recurrent.ran

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.helpers.BackwardHelper
import com.kotlinnlp.simplednn.core.arrays.getInputErrors
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the backward on a [layer].
 *
 * @property layer the [RANLayer] in which the backward is executed
 */
class RANBackwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: RANLayer<InputNDArrayType>
) : BackwardHelper<InputNDArrayType>(layer) {

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param propagateToInput whether to propagate the errors to the input array
   */
  override fun execBackward(propagateToInput: Boolean) {

    val prevStateLayer = this.layer.layerContextWindow.getPrevState() as? RANLayer
    val nextStateLayer = this.layer.layerContextWindow.getNextState() as? RANLayer

    this.layer.applyOutputActivationDeriv() // must be applied BEFORE having added the recurrent contribution

    this.addOutputRecurrentGradients(nextStateLayer = nextStateLayer)

    this.assignGatesGradients(prevStateLayer)

    this.assignParamsGradients(prevStateOutput = prevStateLayer?.outputArray)

    if (propagateToInput) {
      this.assignLayerGradients()
    }
  }

  /**
   * @param prevStateLayer the layer in the previous state
   */
  private fun assignGatesGradients(prevStateLayer: RANLayer<*>?) {

    val gy: DenseNDArray = this.layer.outputArray.errors

    val inputGate = this.layer.inputGate
    val forgetGate = this.layer.forgetGate
    val candidate = this.layer.candidate

    val inG: DenseNDArray = inputGate.values
    val c: DenseNDArray = candidate.values

    val inGDeriv: DenseNDArray = inputGate.calculateActivationDeriv()

    this.layer.inputGate.assignErrorsByProd(c, inGDeriv).assignProd(gy)
    this.layer.candidate.assignErrorsByProd(inG, gy)

    if (prevStateLayer != null) {
      val yPrev: DenseNDArray = prevStateLayer.outputArray.values
      val forGDeriv: DenseNDArray = forgetGate.calculateActivationDeriv()

      this.layer.forgetGate.assignErrorsByProd(yPrev, forGDeriv).assignProd(gy)

    } else {
      this.layer.forgetGate.assignZeroErrors()
    }
  }

  /**
   * @param prevStateOutput the outputArray in the previous state
   */
  private fun assignParamsGradients(prevStateOutput: AugmentedArray<DenseNDArray>?) {

    val x: InputNDArrayType = this.layer.inputArray.values
    val yPrev: DenseNDArray? = prevStateOutput?.valuesNotActivated

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

    this.layer.candidate.assignParamsGradients(
      gw = this.layer.params.candidate.weights.errors.values,
      gb = this.layer.params.candidate.biases.errors.values,
      x = x)
  }

  /**
   *
   */
  private fun assignLayerGradients() {

    this.layer.inputArray
      .assignErrors(this.layer.inputGate.getInputErrors(w = this.layer.params.inputGate.weights.values))
      .assignSum(this.layer.forgetGate.getInputErrors(w = this.layer.params.forgetGate.weights.values))
      .assignSum(this.layer.candidate.getInputErrors(w = this.layer.params.candidate.weights.values))
  }

  /**
   * @param nextStateLayer the layer structure in the next state
   */
  private fun addOutputRecurrentGradients(nextStateLayer: RANLayer<*>?) {

    if (nextStateLayer != null) {
      val gy: DenseNDArray = this.layer.outputArray.errors
      val gyRec: DenseNDArray = this.getLayerRecurrentContribution(nextStateLayer = nextStateLayer)

      gy.assignSum(gyRec)
    }
  }

  /**
   * @param nextStateLayer the layer structure in the next state
   */
  private fun getLayerRecurrentContribution(nextStateLayer: RANLayer<*>): DenseNDArray {

    val gyNext: DenseNDArray = nextStateLayer.outputArray.errors
    val forG: DenseNDArray = nextStateLayer.forgetGate.values

    val gRec1: DenseNDArray = forG.assignProd(gyNext)
    val gRec2: DenseNDArray = nextStateLayer.inputGate.getRecurrentErrors(this.layer.params.inputGate)
    val gRec3: DenseNDArray = nextStateLayer.forgetGate.getRecurrentErrors(this.layer.params.forgetGate)

    return gRec1.assignSum(gRec2).assignSum(gRec3)
  }
}
