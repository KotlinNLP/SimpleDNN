/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.ran

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.BackwardHelper
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the backward on a [layer].
 *
 * @property layer the [RANLayerStructure] in which the backward is executed
 */
class RANBackwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: RANLayerStructure<InputNDArrayType>
) : BackwardHelper<InputNDArrayType> {

  /**
   * A support variable to manage the errors on the parameters during the backward
   */
  lateinit private var paramsErrors: RANLayerParameters

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param paramsErrors the errors of the parameters which will be filled
   * @param propagateToInput whether to propagate the errors to the input array
   */
  override fun backward(paramsErrors: LayerParameters, propagateToInput: Boolean) {

    this.paramsErrors = paramsErrors as RANLayerParameters

    val prevStateLayer = this.layer.layerContextWindow.getPrevStateLayer() as? RANLayerStructure
    val nextStateLayer = this.layer.layerContextWindow.getNextStateLayer() as? RANLayerStructure

    this.layer.applyOutputActivationDeriv() // must be applied BEFORE having added the recurrent contribution

    this.addOutputRecurrentGradients(nextStateLayer)

    this.assignGatesGradients(prevStateLayer)
    this.assignParamsGradients(prevStateLayer?.outputArray)

    if (propagateToInput) {
      this.assignLayerGradients()
    }
  }

  /**
   *
   * @param prevStateLayer the layer in the previous state
   */
  private fun assignGatesGradients(prevStateLayer: RANLayerStructure<*>?) {
    this.layer.params as RANLayerParameters

    val gy: DenseNDArray = this.layer.outputArray.errors

    val inputGate = this.layer.inputGate
    val forgetGate = this.layer.forgetGate
    val candidate = this.layer.candidate

    val inG: DenseNDArray = inputGate.values
    val c: DenseNDArray = candidate.values

    val inGDeriv: DenseNDArray = inputGate.calculateActivationDeriv()

    val gInG: DenseNDArray = this.layer.inputGate.errors
    val gc: DenseNDArray = this.layer.candidate.errors

    gInG.assignProd(c, inGDeriv).assignProd(gy)
    gc.assignProd(inG, gy)

    if (prevStateLayer != null) {
      val yPrev: DenseNDArray = prevStateLayer.outputArray.values
      val forGDeriv: DenseNDArray = forgetGate.calculateActivationDeriv()
      val gForG: DenseNDArray = this.layer.forgetGate.errors

      gForG.assignProd(yPrev, forGDeriv).assignProd(gy)
    }
  }

  /**
   *
   * @param prevStateOutput the outputArray in the previous state
   */
  private fun assignParamsGradients(prevStateOutput: AugmentedArray<DenseNDArray>?) {

    val x: InputNDArrayType = this.layer.inputArray.values
    val yPrev: DenseNDArray? = prevStateOutput?.valuesNotActivated

    this.layer.inputGate.assignParamsGradients(paramsErrors = this.paramsErrors.inputGate, x = x, yPrev = yPrev)
    this.layer.forgetGate.assignParamsGradients(paramsErrors = this.paramsErrors.forgetGate, x = x, yPrev = yPrev)

    val gc: DenseNDArray = this.layer.candidate.errors
    val gwc: NDArray<*> = this.paramsErrors.candidate.weights.values
    val gbc: DenseNDArray = this.paramsErrors.candidate.biases.values
    gwc.assignDot(gc, x.T)
    gbc.assignValues(gc)
  }

  /**
   *
   */
  private fun assignLayerGradients() { this.layer.params as RANLayerParameters

    val wInG: DenseNDArray = this.layer.params.inputGate.weights.values as DenseNDArray
    val wForG: DenseNDArray = this.layer.params.forgetGate.weights.values as DenseNDArray
    val wC: DenseNDArray = this.layer.params.candidate.weights.values as DenseNDArray

    val gInG: DenseNDArray = this.layer.inputGate.errors
    val gForG: DenseNDArray = this.layer.forgetGate.errors
    val gC: DenseNDArray = this.layer.candidate.errors

    this.layer.inputArray
      .assignErrors(gForG.T.dot(wForG))
      .assignSum(gC.T.dot(wC))
      .assignSum(gInG.T.dot(wInG))
  }

  /**
   *
   * @param nextStateLayer the layer structure in the next state
   */
  private fun addOutputRecurrentGradients(nextStateLayer: RANLayerStructure<*>?) {

    if (nextStateLayer != null) {
      val gy: DenseNDArray = this.layer.outputArray.errors
      val gyRec: DenseNDArray = this.getLayerRecurrentContribution(nextStateLayer)

      gy.assignSum(gyRec)
    }
  }

  /**
   *
   * @param nextStateLayer the layer structure in the next state
   */
  private fun getLayerRecurrentContribution(nextStateLayer: RANLayerStructure<*>): DenseNDArray {
    this.layer.params as RANLayerParameters

    val inputGate = nextStateLayer.inputGate
    val forgetGate = nextStateLayer.forgetGate

    val gyNext: DenseNDArray = nextStateLayer.outputArray.errors

    val forG: DenseNDArray = forgetGate.values

    val gInG: DenseNDArray = inputGate.errors
    val gForG: DenseNDArray = forgetGate.errors

    val wrInG: DenseNDArray = this.layer.params.inputGate.recurrentWeights.values
    val wrForG: DenseNDArray = this.layer.params.forgetGate.recurrentWeights.values

    val gRec1: DenseNDArray = forG.assignProd(gyNext)
    val gRec2: DenseNDArray = gInG.T.dot(wrInG)
    val gRec3: DenseNDArray = gForG.T.dot(wrForG)

    return gRec1.assignSum(gRec2).assignSum(gRec3)
  }
}
