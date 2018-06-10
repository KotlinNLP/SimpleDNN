/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.types.recurrent.lstm

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.helpers.BackwardHelper
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the backward on a [layer].
 *
 * @property layer the [LSTMLayerStructure] in which the backward is executed
 */
class LSTMBackwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: LSTMLayerStructure<InputNDArrayType>
) : BackwardHelper<InputNDArrayType> {

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param paramsErrors the errors of the parameters which will be filled
   * @param propagateToInput whether to propagate the errors to the input array
   * @param mePropK the k factor of the 'meProp' algorithm to propagate from the k (in percentage) output nodes with
   *                the top errors (ignored if null)
   */
  override fun backward(paramsErrors: LayerParameters<*>, propagateToInput: Boolean, mePropK: Double?) {

    // TODO: implement 'meProp' algorithm

    val prevStateLayer = this.layer.layerContextWindow.getPrevStateLayer() as? LSTMLayerStructure
    val nextStateLayer = this.layer.layerContextWindow.getNextStateLayer() as? LSTMLayerStructure

    if (nextStateLayer != null) {
      this.addOutputRecurrentGradients(nextStateLayer)
    }

    this.assignGatesGradients(prevStateLayer = prevStateLayer, nextStateLayer = nextStateLayer)

    this.assignParamsGradients(
      paramsErrors = paramsErrors as LSTMLayerParameters,
      prevStateOutput = prevStateLayer?.outputArray)

    if (propagateToInput) {
      this.assignLayerGradients()
    }
  }

  /**
   *
   * @param nextStateLayer the layer structure in the next state
   */
  fun getLayerRecurrentContribution(nextStateLayer: LSTMLayerStructure<*>): DenseNDArray {

    this.layer.params as LSTMLayerParameters

    val gInGNext: DenseNDArray = nextStateLayer.inputGate.errors
    val gOutGNext: DenseNDArray = nextStateLayer.outputGate.errors
    val gForGNext: DenseNDArray = nextStateLayer.forgetGate.errors
    val gCandNext: DenseNDArray = nextStateLayer.candidate.errors

    val wInGRec: DenseNDArray = this.layer.params.inputGate.recurrentWeights.values as DenseNDArray
    val wOutGRec: DenseNDArray = this.layer.params.outputGate.recurrentWeights.values as DenseNDArray
    val wForGRec: DenseNDArray = this.layer.params.forgetGate.recurrentWeights.values as DenseNDArray
    val wCandRec: DenseNDArray = this.layer.params.candidate.recurrentWeights.values as DenseNDArray

    val gRec1: DenseNDArray = gInGNext.t.dot(wInGRec)
    val gRec2: DenseNDArray = gOutGNext.t.dot(wOutGRec)
    val gRec3: DenseNDArray = gForGNext.t.dot(wForGRec)
    val gRec4: DenseNDArray = gCandNext.t.dot(wCandRec)

    return gRec1.assignSum(gRec2).assignSum(gRec3).assignSum(gRec4)
  }

  /**
   *
   * @param prevStateLayer the layer in the previous state
   * @param nextStateLayer the layer in the next state
   */
  private fun assignGatesGradients(prevStateLayer: LSTMLayerStructure<*>?, nextStateLayer: LSTMLayerStructure<*>?) {

    val gy: DenseNDArray = this.layer.outputArray.errors

    val inG: DenseNDArray = this.layer.inputGate.values
    val outG: DenseNDArray = this.layer.outputGate.values
    val cand: DenseNDArray = this.layer.candidate.values
    val cell: DenseNDArray = this.layer.cell.values

    val inGDeriv: DenseNDArray = this.layer.inputGate.calculateActivationDeriv()
    val outGDeriv: DenseNDArray = this.layer.outputGate.calculateActivationDeriv()

    // WARNING: gCell must be calculated before others
    val gCell: DenseNDArray = this.layer.cell.assignErrorsByProd(outG, gy)

    if (this.layer.cell.hasActivation) {
      val cellDeriv: DenseNDArray = this.layer.cell.calculateActivationDeriv()
      this.layer.cell.errors.assignProd(cellDeriv)
    }

    if (nextStateLayer != null) { // add recurrent contribution
      gCell.assignSum(this.getCellRecurrentContribution(nextStateLayer))
    }

    this.layer.outputGate.assignErrorsByProd(cell, outGDeriv).assignProd(gy)
    this.layer.inputGate.assignErrorsByProd(gCell, cand).assignProd(inGDeriv)

    if (prevStateLayer != null) {
      val cellPrev: DenseNDArray = prevStateLayer.cell.valuesNotActivated
      val forGDeriv: DenseNDArray = this.layer.forgetGate.calculateActivationDeriv()
      this.layer.forgetGate.assignErrorsByProd(gCell, cellPrev).assignProd(forGDeriv)

    } else {
      this.layer.forgetGate.assignZeroErrors()
    }

    this.layer.candidate.assignErrorsByProd(gCell, inG)

    if (this.layer.candidate.hasActivation) {
      val candDeriv: DenseNDArray = this.layer.candidate.calculateActivationDeriv()
      this.layer.candidate.errors.assignProd(candDeriv)
    }
  }

  /**
   * @param paramsErrors the errors of the parameters which will be filled
   * @param prevStateOutput the outputArray in the previous state
   */
  private fun assignParamsGradients(paramsErrors: LSTMLayerParameters, prevStateOutput: AugmentedArray<DenseNDArray>?) {

    val x: InputNDArrayType = this.layer.inputArray.values
    val yPrev: DenseNDArray? = prevStateOutput?.values

    this.layer.inputGate.assignParamsGradients(paramsErrors = paramsErrors.inputGate, x = x, yPrev = yPrev)
    this.layer.outputGate.assignParamsGradients(paramsErrors = paramsErrors.outputGate, x = x, yPrev = yPrev)
    this.layer.forgetGate.assignParamsGradients(paramsErrors = paramsErrors.forgetGate, x = x, yPrev = yPrev)
    this.layer.candidate.assignParamsGradients(paramsErrors = paramsErrors.candidate, x = x, yPrev = yPrev)
  }

  /**
   *
   */
  private fun assignLayerGradients() { this.layer.params as LSTMLayerParameters

    val wInG: DenseNDArray = this.layer.params.inputGate.weights.values as DenseNDArray
    val wOutG: DenseNDArray = this.layer.params.outputGate.weights.values as DenseNDArray
    val wForG: DenseNDArray = this.layer.params.forgetGate.weights.values as DenseNDArray
    val wCand: DenseNDArray = this.layer.params.candidate.weights.values as DenseNDArray

    val gInG: DenseNDArray = this.layer.inputGate.errors
    val gOutG: DenseNDArray = this.layer.outputGate.errors
    val gForG: DenseNDArray = this.layer.forgetGate.errors
    val gCand: DenseNDArray = this.layer.candidate.errors

    this.layer.inputArray
      .assignErrorsByDotT(gInG.t, wInG)
      .assignSum(gOutG.t.dot(wOutG))
      .assignSum(gForG.t.dot(wForG))
      .assignSum(gCand.t.dot(wCand))
  }

  /**
   *
   * @param nextStateLayer the layer structure in the next state
   */
  private fun addOutputRecurrentGradients(nextStateLayer: LSTMLayerStructure<*>) {

    val gy: DenseNDArray = this.layer.outputArray.errors
    val gyRec: DenseNDArray = this.getLayerRecurrentContribution(nextStateLayer)

    gy.assignSum(gyRec)
  }

  /**
   *
   * @param nextStateLayer the layer structure in the next state
   */
  private fun getCellRecurrentContribution(nextStateLayer: LSTMLayerStructure<*>): DenseNDArray {

    val gCellNext: DenseNDArray = nextStateLayer.cell.errors
    val forGNext: DenseNDArray = nextStateLayer.forgetGate.values

    return gCellNext.prod(forGNext)
  }
}
