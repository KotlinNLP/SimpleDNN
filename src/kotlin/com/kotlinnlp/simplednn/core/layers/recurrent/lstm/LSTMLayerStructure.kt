/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.lstm

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.activations.Sigmoid
import com.kotlinnlp.simplednn.core.layers.*
import com.kotlinnlp.simplednn.core.layers.recurrent.*
import com.kotlinnlp.simplednn.simplemath.ndarray.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * @param inputArray the input array of the layer
 * @param outputArray the output array of the layer
 * @param params the parameters which connect the input to the output
 * @param layerContextWindow the context window used for the forward and the backward
 * @param activationFunction the activation function of the layer
 * @param dropout The probability of dropout (default 0.0).
 *                If applying it, the usual value is 0.5 (better 0.25 if it's the first layer).
 */
class LSTMLayerStructure<InputNDArrayType : NDArray<InputNDArrayType>>(
  inputArray: AugmentedArray<InputNDArrayType>,
  outputArray: AugmentedArray<DenseNDArray>,
  params: LayerParameters,
  layerContextWindow: LayerContextWindow,
  activationFunction: ActivationFunction? = null,
  dropout: Double = 0.0
) : RecurrentLayerStructure<InputNDArrayType>(
  inputArray = inputArray,
  outputArray = outputArray,
  params = params,
  layerContextWindow = layerContextWindow,
  activationFunction = activationFunction,
  dropout = dropout) {

  /**
   * A support variable to manage the errors on the parameters during the backward
   */
  var paramsErrors: LSTMLayerParameters? = null

  /**
   *
   */
  val inputGate = GateUnit<InputNDArrayType>(outputArray.size)

  /**
   *
   */
  val outputGate = GateUnit<InputNDArrayType>(outputArray.size)

  /**
   *
   */
  val forgetGate = GateUnit<InputNDArrayType>(outputArray.size)

  /**
   *
   */
  val candidate = GateUnit<InputNDArrayType>(outputArray.size)

  /**
   *
   */
  val cell = AugmentedArray<DenseNDArray>(outputArray.size)

  /**
   * Initialization: set the activation function of the gates
   */
  init {

    this.inputGate.setActivation(Sigmoid())
    this.outputGate.setActivation(Sigmoid())
    this.forgetGate.setActivation(Sigmoid())

    if (activationFunction != null) {
      this.candidate.setActivation(activationFunction)
      this.cell.setActivation(activationFunction)
    }
  }

  /**
   * y = outG * f(cell)
   */
  override fun forwardInput() {

    this.setGates(this.layerContextWindow.getPrevStateLayer()) // must be called before accessing to the activated values of the gates

    val y: DenseNDArray = this.outputArray.values
    val outG: DenseNDArray = this.outputGate.values
    val cellA: DenseNDArray = this.cell.values

    y.assignProd(outG, cellA)
  }

  /**
   * Set gates values
   *
   * inG = sigmoid(wIn (dot) x + bIn + wInRec (dot) yPrev)
   * outG = sigmoid(wOut (dot) x + bOut + wOutRec (dot) yPrev)
   * forG = sigmoid(wFor (dot) x + bFor + wForRec (dot) yPrev)
   * cand = f(wCand (dot) x + bC + wCandRec (dot) yPrev)
   * cell = inG * cand + forG * cellPrev
   */
  private fun setGates(prevStateLayer: LayerStructure<*>?) {

    this.forwardGates()

    if (prevStateLayer != null) {
      this.addGatesRecurrentContribute(prevStateLayer)
    }

    this.activateGates()

    val cell: DenseNDArray = this.cell.values
    val inG: DenseNDArray = this.inputGate.values
    val cand: DenseNDArray = this.candidate.values
    cell.assignProd(inG, cand)

    if (prevStateLayer != null) { // add recurrent contribute to the cell
      val forG: DenseNDArray = this.forgetGate.values
      val cellPrev: DenseNDArray = (prevStateLayer as LSTMLayerStructure).cell.valuesNotActivated
      cell.assignSum(forG.prod(cellPrev))
    }

    this.cell.activate()
  }

  /**
   *
   */
  private fun forwardGates() { this.params as LSTMLayerParameters

    val x: InputNDArrayType = this.inputArray.values

    this.inputGate.forward(this.params.inputGate, x)
    this.outputGate.forward(this.params.outputGate, x)
    this.forgetGate.forward(this.params.forgetGate, x)
    this.candidate.forward(this.params.candidate, x)
  }

  /**
   *
   */
  private fun addGatesRecurrentContribute(prevStateLayer: LayerStructure<*>) {
    this.params as LSTMLayerParameters

    val yPrev: DenseNDArray = prevStateLayer.outputArray.values

    this.inputGate.addRecurrentContribute(this.params.inputGate, yPrev)
    this.outputGate.addRecurrentContribute(this.params.outputGate, yPrev)
    this.forgetGate.addRecurrentContribute(this.params.forgetGate, yPrev)
    this.candidate.addRecurrentContribute(this.params.candidate, yPrev)
  }

  /**
   *
   */
  private fun activateGates() {
    this.inputGate.activate()
    this.outputGate.activate()
    this.forgetGate.activate()
    this.candidate.activate()
  }

  /**
   *
   */
  override fun backward(paramsErrors: LayerParameters, propagateToInput: Boolean) {

    this.paramsErrors = paramsErrors as LSTMLayerParameters

    val prevStateLayer = this.layerContextWindow.getPrevStateLayer() as? LSTMLayerStructure
    val nextStateLayer = this.layerContextWindow.getNextStateLayer() as? LSTMLayerStructure

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

    val gy: DenseNDArray = this.outputArray.errors

    val inG: DenseNDArray = this.inputGate.values
    val outG: DenseNDArray = this.outputGate.values
    val cand: DenseNDArray = this.candidate.values
    val cell: DenseNDArray = this.cell.values

    val inGDeriv: DenseNDArray = this.inputGate.calculateActivationDeriv()
    val outGDeriv: DenseNDArray = this.outputGate.calculateActivationDeriv()
    val candDeriv: DenseNDArray = this.candidate.calculateActivationDeriv()
    val cellDeriv: DenseNDArray = this.cell.calculateActivationDeriv()

    val gInG: DenseNDArray = this.inputGate.errors
    val gOutG: DenseNDArray = this.outputGate.errors
    val gForG: DenseNDArray = this.forgetGate.errors
    val gCand: DenseNDArray = this.candidate.errors
    val gCell: DenseNDArray = this.cell.errors

    gCell.assignProd(outG, cellDeriv).assignProd(gy) // attention: must be calculated before others

    if (nextStateLayer != null) { // add recurrent contribute to gCell
      val forGNext: DenseNDArray = nextStateLayer.forgetGate.values
      val gCellNext: DenseNDArray = nextStateLayer.cell.errors
      gCell.assignSum(gCellNext.prod(forGNext))
    }

    gOutG.assignProd(cell, outGDeriv).assignProd(gy)

    gInG.assignProd(gCell, cand).assignProd(inGDeriv)

    if (prevStateLayer != null) {
      val cellPrev: DenseNDArray = prevStateLayer.cell.valuesNotActivated
      val forGDeriv: DenseNDArray = this.forgetGate.calculateActivationDeriv()
      gForG.assignProd(gCell, cellPrev).assignProd(forGDeriv)
    }

    gCand.assignProd(gCell, inG).assignProd(candDeriv)
  }

  /**
   *
   * @param prevStateOutput the outputArray in the previous state
   */
  private fun assignParamsGradients(prevStateOutput: AugmentedArray<DenseNDArray>?) {

    val x: InputNDArrayType = this.inputArray.values
    val yPrev: DenseNDArray? = prevStateOutput?.values

    this.inputGate.assignParamsGradients(paramsErrors = this.paramsErrors!!.inputGate, x = x, yPrev = yPrev)
    this.outputGate.assignParamsGradients(paramsErrors = this.paramsErrors!!.outputGate, x = x, yPrev = yPrev)
    this.forgetGate.assignParamsGradients(paramsErrors = this.paramsErrors!!.forgetGate, x = x, yPrev = yPrev)
    this.candidate.assignParamsGradients(paramsErrors = this.paramsErrors!!.candidate, x = x, yPrev = yPrev)
  }

  /**
   *
   */
  private fun assignLayerGradients() { this.params as LSTMLayerParameters

    val gx: DenseNDArray = this.inputArray.errors

    val wInG: DenseNDArray = this.params.inputGate.weights.values as DenseNDArray
    val wOutG: DenseNDArray = this.params.outputGate.weights.values as DenseNDArray
    val wForG: DenseNDArray = this.params.forgetGate.weights.values as DenseNDArray
    val wCand: DenseNDArray = this.params.candidate.weights.values as DenseNDArray

    val gInG: DenseNDArray = this.inputGate.errors
    val gOutG: DenseNDArray = this.outputGate.errors
    val gForG: DenseNDArray = this.forgetGate.errors
    val gCand: DenseNDArray = this.candidate.errors

    gx.assignValues(gInG.T.dot(wInG))
      .assignSum(gOutG.T.dot(wOutG))
      .assignSum(gForG.T.dot(wForG))
      .assignSum(gCand.T.dot(wCand))

    if (this.inputArray.hasActivation && gx is DenseNDArray) {
      gx.assignProd(this.inputArray.calculateActivationDeriv())
    }
  }

  /**
   *
   * @param nextStateLayer the layer structure in the next state
   */
  private fun addOutputRecurrentGradients(nextStateLayer: LSTMLayerStructure<*>?) {

    if (nextStateLayer != null) {
      val gy: DenseNDArray = this.outputArray.errors
      val gCell: DenseNDArray = this.cell.errors
      val gyRec: DenseNDArray = this.getLayerRecurrentContribute(nextStateLayer)
      val gCellRec: DenseNDArray = this.getCellRecurrentContribute(nextStateLayer)

      gy.assignSum(gyRec)
      gCell.assignSum(gCellRec)
    }
  }

  /**
   *
   * @param nextStateLayer the layer structure in the next state
   */
  private fun getLayerRecurrentContribute(nextStateLayer: LSTMLayerStructure<*>): DenseNDArray {
    this.params as LSTMLayerParameters

    val gInGNext: DenseNDArray = nextStateLayer.inputGate.errors
    val gOutGNext: DenseNDArray = nextStateLayer.outputGate.errors
    val gForGNext: DenseNDArray = nextStateLayer.forgetGate.errors
    val gCandNext: DenseNDArray = nextStateLayer.candidate.errors

    val wInGRec: DenseNDArray = this.params.inputGate.recurrentWeights.values
    val wOutGRec: DenseNDArray = this.params.outputGate.recurrentWeights.values
    val wForGRec: DenseNDArray = this.params.forgetGate.recurrentWeights.values
    val wCandRec: DenseNDArray = this.params.candidate.recurrentWeights.values

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
  private fun getCellRecurrentContribute(nextStateLayer: LSTMLayerStructure<*>): DenseNDArray {

    val gCellNext: DenseNDArray = nextStateLayer.cell.errors
    val forGNext: DenseNDArray = nextStateLayer.forgetGate.values

    return gCellNext.prod(forGNext)
  }
}
