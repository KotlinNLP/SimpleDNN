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
import com.kotlinnlp.simplednn.simplemath.NDArray

/**
 * @param inputArray the input array of the layer
 * @param outputArray the output array of the layer
 * @param params the parameters which connect the input to the output
 * @param layerContextWindow the context window used for the forward and the backward
 * @param activationFunction the activation function of the layer
 * @param dropout The probability of dropout (default 0.0).
 *                If applying it, the usual value is 0.5 (better 0.25 if it's the first layer).
 */
class LSTMLayerStructure(
  inputArray: AugmentedArray,
  outputArray: AugmentedArray,
  params: LayerParameters,
  layerContextWindow: LayerContextWindow,
  activationFunction: ActivationFunction? = null,
  dropout: Double = 0.0
) : RecurrentLayerStructure(
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
  val inputGate = GateUnit(outputArray.size)

  /**
   *
   */
  val outputGate = GateUnit(outputArray.size)

  /**
   *
   */
  val forgetGate = GateUnit(outputArray.size)

  /**
   *
   */
  val candidate = GateUnit(outputArray.size)

  /**
   *
   */
  val cell = AugmentedArray(outputArray.size)

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

    val y = this.outputArray.values
    val outG = this.outputGate.values
    val cellA = this.cell.values

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
  private fun setGates(prevStateLayer: LayerStructure?) {

    this.forwardGates()

    if (prevStateLayer != null) {
      this.addGatesRecurrentContribute(prevStateLayer)
    }

    this.activateGates()

    val cell = this.cell.values
    val inG = this.inputGate.values
    val cand = this.candidate.values
    cell.assignProd(inG, cand)

    if (prevStateLayer != null) { // add recurrent contribute to the cell
      val forG = this.forgetGate.values
      val cellPrev = (prevStateLayer as LSTMLayerStructure).cell.valuesNotActivated
      cell.assignSum(forG.prod(cellPrev))
    }

    this.cell.activate()
  }

  /**
   *
   */
  private fun forwardGates() { this.params as LSTMLayerParameters

    val x = this.inputArray.values

    this.inputGate.forward(this.params.inputGate, x)
    this.outputGate.forward(this.params.outputGate, x)
    this.forgetGate.forward(this.params.forgetGate, x)
    this.candidate.forward(this.params.candidate, x)
  }

  /**
   *
   */
  private fun addGatesRecurrentContribute(prevStateLayer: LayerStructure) {
    this.params as LSTMLayerParameters

    val yPrev = prevStateLayer.outputArray.values

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
  private fun assignGatesGradients(prevStateLayer: LSTMLayerStructure?,
                                   nextStateLayer: LSTMLayerStructure?) {

    val gy = this.outputArray.errors

    val inG = this.inputGate.values
    val outG = this.outputGate.values
    val cand = this.candidate.values
    val cell = this.cell.values

    val inGDeriv = this.inputGate.calculateActivationDeriv()
    val outGDeriv = this.outputGate.calculateActivationDeriv()
    val candDeriv = this.candidate.calculateActivationDeriv()
    val cellDeriv = this.cell.calculateActivationDeriv()

    val gInG = this.inputGate.errors
    val gOutG = this.outputGate.errors
    val gForG = this.forgetGate.errors
    val gCand = this.candidate.errors
    val gCell = this.cell.errors

    gCell.assignProd(outG, cellDeriv).assignProd(gy) // attention: must be calculated before others

    if (nextStateLayer != null) { // add recurrent contribute to gCell
      val forGNext = nextStateLayer.forgetGate.values
      val gCellNext = nextStateLayer.cell.errors
      gCell.assignSum(gCellNext.prod(forGNext))
    }

    gOutG.assignProd(cell, outGDeriv).assignProd(gy)

    gInG.assignProd(gCell, cand).assignProd(inGDeriv)

    if (prevStateLayer != null) {
      val cellPrev = prevStateLayer.cell.valuesNotActivated
      val forGDeriv = this.forgetGate.calculateActivationDeriv()
      gForG.assignProd(gCell, cellPrev).assignProd(forGDeriv)
    }

    gCand.assignProd(gCell, inG).assignProd(candDeriv)
  }

  /**
   *
   * @param prevStateOutput the outputArray in the previous state
   */
  private fun assignParamsGradients(prevStateOutput: AugmentedArray?) {

    val x = this.inputArray.values
    val yPrev = prevStateOutput?.values

    this.setGateParamsGradients(this.inputGate, this.paramsErrors!!.inputGate, x, yPrev)
    this.setGateParamsGradients(this.outputGate, this.paramsErrors!!.outputGate, x, yPrev)
    this.setGateParamsGradients(this.forgetGate, this.paramsErrors!!.forgetGate, x, yPrev)
    this.setGateParamsGradients(this.candidate, this.paramsErrors!!.candidate, x, yPrev)
  }

  /**
   *
   * @param gate the gate unit
   * @param gateParams the gate unit parameters
   * @param x the input NDArray of the gate
   * @param yPrev the output NDArray of the gate in the previous state
   *
   * gb = gGate * 1
   * gw = gGate (dot) x
   * gwRec = gGate (dot) yPrev
   */
  private fun setGateParamsGradients(gate: GateUnit,
                                     gateParams: GateParametersUnit,
                                     x: NDArray,
                                     yPrev: NDArray?) {

    val gGate = gate.errors
    val gb = gateParams.biases.values
    val gw = gateParams.weights.values
    val gwRec = gateParams.recurrentWeights.values

    gb.assignValues(gGate)
    gw.assignDot(gGate, x.T)

    if (yPrev != null) {
      gwRec.assignDot(gGate, yPrev.T)
    }
  }

  /**
   *
   */
  private fun assignLayerGradients() { this.params as LSTMLayerParameters

    val gx = this.inputArray.errors

    val wInG = this.params.inputGate.weights.values
    val wOutG = this.params.outputGate.weights.values
    val wForG = this.params.forgetGate.weights.values
    val wCand = this.params.candidate.weights.values

    val gInG = this.inputGate.errors
    val gOutG = this.outputGate.errors
    val gForG = this.forgetGate.errors
    val gCand = this.candidate.errors

    gx.assignValues(gInG.T.dot(wInG))
      .assignSum(gOutG.T.dot(wOutG))
      .assignSum(gForG.T.dot(wForG))
      .assignSum(gCand.T.dot(wCand))
  }

  /**
   *
   * @param nextStateLayer the layer structure in the next state
   */
  private fun addOutputRecurrentGradients(nextStateLayer: LSTMLayerStructure?) {

    if (nextStateLayer != null) {
      val gy = this.outputArray.errors
      val gCell = this.cell.errors
      val gyRec = this.getLayerRecurrentContribute(nextStateLayer)
      val gCellRec = this.getCellRecurrentContribute(nextStateLayer)

      gy.assignSum(gyRec)
      gCell.assignSum(gCellRec)
    }
  }

  /**
   *
   * @param nextStateLayer the layer structure in the next state
   */
  private fun getLayerRecurrentContribute(nextStateLayer: LSTMLayerStructure): NDArray {
    this.params as LSTMLayerParameters

    val gInGNext: NDArray = nextStateLayer.inputGate.errors
    val gOutGNext: NDArray = nextStateLayer.outputGate.errors
    val gForGNext: NDArray = nextStateLayer.forgetGate.errors
    val gCandNext: NDArray = nextStateLayer.candidate.errors

    val wInGRec: NDArray = this.params.inputGate.recurrentWeights.values
    val wOutGRec: NDArray = this.params.outputGate.recurrentWeights.values
    val wForGRec: NDArray = this.params.forgetGate.recurrentWeights.values
    val wCandRec: NDArray = this.params.candidate.recurrentWeights.values

    val gRec1 = gInGNext.T.dot(wInGRec)
    val gRec2 = gOutGNext.T.dot(wOutGRec)
    val gRec3 = gForGNext.T.dot(wForGRec)
    val gRec4 = gCandNext.T.dot(wCandRec)

    return gRec1.assignSum(gRec2).assignSum(gRec3).assignSum(gRec4)
  }

  /**
   *
   * @param nextStateLayer the layer structure in the next state
   */
  private fun getCellRecurrentContribute(nextStateLayer: LSTMLayerStructure): NDArray {

    val gCellNext = nextStateLayer.cell.errors
    val forGNext = nextStateLayer.forgetGate.values

    return gCellNext.prod(forGNext)
  }
}
