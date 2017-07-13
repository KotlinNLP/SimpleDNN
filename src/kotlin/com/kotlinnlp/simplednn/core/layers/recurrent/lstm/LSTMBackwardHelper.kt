/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.lstm

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.BackwardHelper
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
   * A support variable to manage the errors on the parameters during the backward
   */
  lateinit private var paramsErrors: LSTMLayerParameters

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param paramsErrors the errors of the parameters which will be filled
   * @param propagateToInput whether to propagate the errors to the input array
   */
  override fun backward(paramsErrors: LayerParameters, propagateToInput: Boolean) {

    this.paramsErrors = paramsErrors as LSTMLayerParameters

    val prevStateLayer = this.layer.layerContextWindow.getPrevStateLayer() as? LSTMLayerStructure
    val nextStateLayer = this.layer.layerContextWindow.getNextStateLayer() as? LSTMLayerStructure

    this.addOutputRecurrentGradients(nextStateLayer)

    this.assignGatesGradients(prevStateLayer = prevStateLayer, nextStateLayer = nextStateLayer)
    this.assignParamsGradients(prevStateOutput = prevStateLayer?.outputArray)

    if (propagateToInput) {
      this.assignLayerGradients()
    }
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
    val candDeriv: DenseNDArray = this.layer.candidate.calculateActivationDeriv()
    val cellDeriv: DenseNDArray = this.layer.cell.calculateActivationDeriv()

    val gInG: DenseNDArray = this.layer.inputGate.errors
    val gOutG: DenseNDArray = this.layer.outputGate.errors
    val gForG: DenseNDArray = this.layer.forgetGate.errors
    val gCand: DenseNDArray = this.layer.candidate.errors
    val gCell: DenseNDArray = this.layer.cell.errors

    gCell.assignProd(outG, cellDeriv).assignProd(gy) // attention: must be calculated before others

    if (nextStateLayer != null) { // add recurrent contribution to gCell
      val forGNext: DenseNDArray = nextStateLayer.forgetGate.values
      val gCellNext: DenseNDArray = nextStateLayer.cell.errors
      gCell.assignSum(gCellNext.prod(forGNext))
    }

    gOutG.assignProd(cell, outGDeriv).assignProd(gy)

    gInG.assignProd(gCell, cand).assignProd(inGDeriv)

    if (prevStateLayer != null) {
      val cellPrev: DenseNDArray = prevStateLayer.cell.valuesNotActivated
      val forGDeriv: DenseNDArray = this.layer.forgetGate.calculateActivationDeriv()
      gForG.assignProd(gCell, cellPrev).assignProd(forGDeriv)
    }

    gCand.assignProd(gCell, inG).assignProd(candDeriv)
  }

  /**
   *
   * @param prevStateOutput the outputArray in the previous state
   */
  private fun assignParamsGradients(prevStateOutput: AugmentedArray<DenseNDArray>?) {

    val x: InputNDArrayType = this.layer.inputArray.values
    val yPrev: DenseNDArray? = prevStateOutput?.values

    this.layer.inputGate.assignParamsGradients(paramsErrors = this.paramsErrors.inputGate, x = x, yPrev = yPrev)
    this.layer.outputGate.assignParamsGradients(paramsErrors = this.paramsErrors.outputGate, x = x, yPrev = yPrev)
    this.layer.forgetGate.assignParamsGradients(paramsErrors = this.paramsErrors.forgetGate, x = x, yPrev = yPrev)
    this.layer.candidate.assignParamsGradients(paramsErrors = this.paramsErrors.candidate, x = x, yPrev = yPrev)
  }

  /**
   *
   */
  private fun assignLayerGradients() { this.layer.params as LSTMLayerParameters

    val gx: DenseNDArray = this.layer.inputArray.errors

    val wInG: DenseNDArray = this.layer.params.inputGate.weights.values as DenseNDArray
    val wOutG: DenseNDArray = this.layer.params.outputGate.weights.values as DenseNDArray
    val wForG: DenseNDArray = this.layer.params.forgetGate.weights.values as DenseNDArray
    val wCand: DenseNDArray = this.layer.params.candidate.weights.values as DenseNDArray

    val gInG: DenseNDArray = this.layer.inputGate.errors
    val gOutG: DenseNDArray = this.layer.outputGate.errors
    val gForG: DenseNDArray = this.layer.forgetGate.errors
    val gCand: DenseNDArray = this.layer.candidate.errors

    gx.assignValues(gInG.T.dot(wInG))
      .assignSum(gOutG.T.dot(wOutG))
      .assignSum(gForG.T.dot(wForG))
      .assignSum(gCand.T.dot(wCand))
  }

  /**
   *
   * @param nextStateLayer the layer structure in the next state
   */
  private fun addOutputRecurrentGradients(nextStateLayer: LSTMLayerStructure<*>?) {

    if (nextStateLayer != null) {
      val gy: DenseNDArray = this.layer.outputArray.errors
      val gCell: DenseNDArray = this.layer.cell.errors
      val gyRec: DenseNDArray = this.getLayerRecurrentContribution(nextStateLayer)
      val gCellRec: DenseNDArray = this.getCellRecurrentContribution(nextStateLayer)

      gy.assignSum(gyRec)
      gCell.assignSum(gCellRec)
    }
  }

  /**
   *
   * @param nextStateLayer the layer structure in the next state
   */
  private fun getLayerRecurrentContribution(nextStateLayer: LSTMLayerStructure<*>): DenseNDArray {
    this.layer.params as LSTMLayerParameters

    val gInGNext: DenseNDArray = nextStateLayer.inputGate.errors
    val gOutGNext: DenseNDArray = nextStateLayer.outputGate.errors
    val gForGNext: DenseNDArray = nextStateLayer.forgetGate.errors
    val gCandNext: DenseNDArray = nextStateLayer.candidate.errors

    val wInGRec: DenseNDArray = this.layer.params.inputGate.recurrentWeights.values
    val wOutGRec: DenseNDArray = this.layer.params.outputGate.recurrentWeights.values
    val wForGRec: DenseNDArray = this.layer.params.forgetGate.recurrentWeights.values
    val wCandRec: DenseNDArray = this.layer.params.candidate.recurrentWeights.values

    val gRec1: DenseNDArray = gInGNext.T.dot(wInGRec)
    val gRec2: DenseNDArray = gOutGNext.T.dot(wOutGRec)
    val gRec3: DenseNDArray = gForGNext.T.dot(wForGRec)
    val gRec4: DenseNDArray = gCandNext.T.dot(wCandRec)

    return gRec1.assignSum(gRec2).assignSum(gRec3).assignSum(gRec4)
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
