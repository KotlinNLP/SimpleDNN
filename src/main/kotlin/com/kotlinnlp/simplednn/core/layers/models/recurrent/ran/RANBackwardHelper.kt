/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.recurrent.ran

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.helpers.BackwardHelper
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
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param paramsErrors the errors of the parameters which will be filled
   * @param propagateToInput whether to propagate the errors to the input array
   */
  override fun backward(paramsErrors: LayerParameters<*>, propagateToInput: Boolean) {

    val prevStateLayer = this.layer.layerContextWindow.getPrevStateLayer() as? RANLayerStructure
    val nextStateLayer = this.layer.layerContextWindow.getNextStateLayer() as? RANLayerStructure

    this.layer.applyOutputActivationDeriv() // must be applied BEFORE having added the recurrent contribution

    this.addOutputRecurrentGradients(nextStateLayer = nextStateLayer)

    this.assignGatesGradients(prevStateLayer)

    this.assignParamsGradients(
      paramsErrors = paramsErrors as RANLayerParameters,
      prevStateOutput = prevStateLayer?.outputArray)

    if (propagateToInput) {
      this.assignLayerGradients()
    }
  }

  /**
   * @param prevStateLayer the layer in the previous state
   */
  private fun assignGatesGradients(prevStateLayer: RANLayerStructure<*>?) {

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
   * @param paramsErrors the errors of the parameters which will be filled
   * @param prevStateOutput the outputArray in the previous state
   */
  private fun assignParamsGradients(paramsErrors: RANLayerParameters,
                                    prevStateOutput: AugmentedArray<DenseNDArray>?) {

    val x: InputNDArrayType = this.layer.inputArray.values
    val yPrev: DenseNDArray? = prevStateOutput?.valuesNotActivated

    this.layer.inputGate.assignParamsGradients(paramsErrors.inputGate, x = x, yPrev = yPrev)
    this.layer.forgetGate.assignParamsGradients(paramsErrors.forgetGate, x = x, yPrev = yPrev)
    this.layer.candidate.assignParamsGradients(paramsErrors.candidate, x = x)
  }

  /**
   *
   */
  private fun assignLayerGradients() {

    this.layer.params as RANLayerParameters

    this.layer.inputArray
      .assignErrors(this.layer.inputGate.getInputErrors(this.layer.params.inputGate))
      .assignSum(this.layer.forgetGate.getInputErrors(this.layer.params.forgetGate))
      .assignSum(this.layer.candidate.getInputErrors(this.layer.params.candidate))
  }

  /**
   * @param nextStateLayer the layer structure in the next state
   */
  private fun addOutputRecurrentGradients(nextStateLayer: RANLayerStructure<*>?) {

    if (nextStateLayer != null) {
      val gy: DenseNDArray = this.layer.outputArray.errors
      val gyRec: DenseNDArray = this.getLayerRecurrentContribution(nextStateLayer = nextStateLayer)

      gy.assignSum(gyRec)
    }
  }

  /**
   * @param nextStateLayer the layer structure in the next state
   */
  private fun getLayerRecurrentContribution(nextStateLayer: RANLayerStructure<*>): DenseNDArray {

    this.layer.params as RANLayerParameters

    val gyNext: DenseNDArray = nextStateLayer.outputArray.errors
    val forG: DenseNDArray = nextStateLayer.forgetGate.values

    val gRec1: DenseNDArray = forG.assignProd(gyNext)
    val gRec2: DenseNDArray = nextStateLayer.inputGate.getRecurrentErrors(this.layer.params.inputGate)
    val gRec3: DenseNDArray = nextStateLayer.forgetGate.getRecurrentErrors(this.layer.params.forgetGate)

    return gRec1.assignSum(gRec2).assignSum(gRec3)
  }
}
