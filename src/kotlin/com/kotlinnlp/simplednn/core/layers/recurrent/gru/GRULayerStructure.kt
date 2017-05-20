/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.gru

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
class GRULayerStructure(
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
  var paramsErrors: GRULayerParameters? = null

  /**
   *
   */
  val candidate = GateUnit(outputArray.size)

  /**
   *
   */
  val resetGate = GateUnit(outputArray.size)

  /**
   *
   */
  val partitionGate = GateUnit(outputArray.size)

  /**
   * Initialization: set the activation function of the gates
   */
  init {

    this.resetGate.setActivation(Sigmoid())
    this.partitionGate.setActivation(Sigmoid())

    if (activationFunction != null) {
      this.candidate.setActivation(activationFunction)
    }
  }

  /**
   * y = p * c + (1 - p) * yPrev
   */
  override fun forwardInput() {

    val prevStateLayer = this.layerContextWindow.getPrevStateLayer()

    this.setGates(prevStateLayer) // must be called before accessing to the activated values of the gates

    val y = this.outputArray.values
    val c = this.candidate.values
    val p = this.partitionGate.values

    // y = p * c
    y.assignProd(p, c)

    // y += (1 - p) * yPrev
    if (prevStateLayer != null) {
      val yPrev: NDArray = prevStateLayer.outputArray.values
      y.assignSum(p.reverseSub(1).prod(yPrev))
    }
  }

  /**
   * Set gates values
   *
   * r = sigmoid(wr (dot) x + br + wrRec (dot) yPrev)
   * p = sigmoid(wp (dot) x + bp + wpRec (dot) yPrev)
   * c = f(wc (dot) x + bc + wcRec (dot) (yPrev * r))
   */
  private fun setGates(prevStateLayer: LayerStructure?) { this.params as GRULayerParameters

    val x = this.inputArray.values

    this.resetGate.forward(this.params.resetGate, x)
    this.partitionGate.forward(this.params.partitionGate, x)
    this.candidate.forward(this.params.candidate, x)

    if (prevStateLayer != null) { // recurrent contribute for r and p
      val yPrev = prevStateLayer.outputArray.values
      this.resetGate.addRecurrentContribute(this.params.resetGate, yPrev)
      this.partitionGate.addRecurrentContribute(this.params.partitionGate, yPrev)
    }

    this.resetGate.activate()
    this.partitionGate.activate()

    if (prevStateLayer != null) { // recurrent contribute for c
      val yPrev = prevStateLayer.outputArray.values
      val r = this.resetGate.values
      this.candidate.addRecurrentContribute(this.params.candidate, r.prod(yPrev))
    }

    this.candidate.activate()
  }

  /**
   *
   */
  override fun backward(paramsErrors: LayerParameters, propagateToInput: Boolean) {

    this.paramsErrors = paramsErrors as GRULayerParameters

    val prevStateOutput = this.layerContextWindow.getPrevStateLayer()?.outputArray
    val nextStateLayer = this.layerContextWindow.getNextStateLayer()

    this.addOutputRecurrentGradients(nextStateLayer as? GRULayerStructure)

    this.assignGatesGradients(prevStateOutput)
    this.assignParamsGradients(prevStateOutput)

    if (propagateToInput) {
      this.assignLayerGradients()
    }
  }

  /**
   *
   * @param prevStateOutput the outputArray in the previous state
   */
  private fun assignGatesGradients(prevStateOutput: AugmentedArray?) { this.params as GRULayerParameters

    val gy = this.outputArray.errors

    val resetGate = this.resetGate
    val partitionGate = this.partitionGate
    val candidate = this.candidate

    val p = partitionGate.values
    val c = candidate.values

    val rDeriv = resetGate.calculateActivationDeriv()
    val pDeriv = partitionGate.calculateActivationDeriv()
    val cDeriv = candidate.calculateActivationDeriv()

    val gr = this.resetGate.errors
    val gp = this.partitionGate.errors
    val gc = this.candidate.errors

    gc.assignProd(p, cDeriv).assignProd(gy)  // gc must be calculated before gr and gp

    if (prevStateOutput == null) {
      gr.zeros()
      gp.assignProd(c, pDeriv).assignProd(gy)

    } else { // recurrent contribute

      val yPrev = prevStateOutput.values
      val wcr = this.params.candidate.recurrentWeights.values

      gr.assignValues(gc.T.dot(wcr)).assignProd(rDeriv).assignProd(yPrev)
      gp.assignProd(c.sub(yPrev), pDeriv).assignProd(gy)
    }
  }

  /**
   *
   * @param prevStateOutput the outputArray in the previous state
   */
  private fun assignParamsGradients(prevStateOutput: AugmentedArray?) {

    val x = this.inputArray.values
    val yPrev = prevStateOutput?.values

    this.setGateParamsGradients(this.resetGate, this.paramsErrors!!.resetGate, x, yPrev = yPrev)
    this.setGateParamsGradients(this.partitionGate, this.paramsErrors!!.partitionGate, x, yPrev = yPrev)
    this.setGateParamsGradients(this.candidate, this.paramsErrors!!.candidate, x)

    if (yPrev != null) { // add recurrent contribute to the recurrent weights of the candidate
      val r = this.resetGate.values
      val gwcr = this.paramsErrors!!.candidate.recurrentWeights.values
      val gc = this.candidate.errors
      gwcr.assignDot(gc, r.prod(yPrev).T)
    }
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
                                     yPrev: NDArray? = null) {

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
  private fun assignLayerGradients() { this.params as GRULayerParameters

    val gx = this.inputArray.errors

    val wp = this.params.partitionGate.weights.values
    val wc = this.params.candidate.weights.values
    val wr = this.params.resetGate.weights.values

    val gp = this.partitionGate.errors
    val gc = this.candidate.errors
    val gr = this.resetGate.errors

    gx.assignValues(gp.T.dot(wp)).assignSum(gc.T.dot(wc)).assignSum(gr.T.dot(wr))
  }

  /**
   *
   * @param nextStateLayer the layer structure in the next state
   */
  private fun addOutputRecurrentGradients(nextStateLayer: GRULayerStructure?) {

    if (nextStateLayer != null) {
      val gy = this.outputArray.errors
      val gyRec = this.getLayerRecurrentContribute(nextStateLayer)

      gy.assignSum(gyRec)
    }
  }

  /**
   *
   * @param nextStateLayer the layer structure in the next state
   */
  private fun getLayerRecurrentContribute(nextStateLayer: GRULayerStructure): NDArray {
    this.params as GRULayerParameters

    val resetGate = nextStateLayer.resetGate
    val partitionGate = nextStateLayer.partitionGate
    val candidate = nextStateLayer.candidate

    val gy = nextStateLayer.outputArray.errors

    val r = resetGate.values
    val p = partitionGate.values

    val gr = resetGate.errors
    val gp = partitionGate.errors
    val gc = candidate.errors

    val wrr = this.params.resetGate.recurrentWeights.values
    val wpr = this.params.partitionGate.recurrentWeights.values
    val wcr = this.params.candidate.recurrentWeights.values

    val gRec1 = gr.T.dot(wrr)
    val gRec2 = gp.T.dot(wpr)
    val gRec3 = gc.T.dot(wcr).prod(r)
    val gRec4 = p.reverseSub(1).prod(gy).T

    return gRec1.assignSum(gRec2).assignSum(gRec3).assignSum(gRec4)
  }
}
