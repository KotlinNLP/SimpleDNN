/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.cfn

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
class CFNLayerStructure(
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
  var paramsErrors: CFNLayerParameters? = null

  /**
   *
   */
  val candidate = AugmentedArray(outputArray.size)

  /**
   *
   */
  val inputGate = GateUnit(outputArray.size)

  /**
   *
   */
  val forgetGate = GateUnit(outputArray.size)

  /**
   *
   */
  var activatedPrevOutput: NDArray? = null

  /**
   * Initialization: set the activation function of the gates
   */
  init {

    this.inputGate.setActivation(Sigmoid())
    this.forgetGate.setActivation(Sigmoid())

    if (this.activationFunction != null) {
      this.candidate.setActivation(this.activationFunction)
    }
  }

  /**
   * y = inG * c + f(yPrev) * forG
   */
  override fun forwardInput() {

    val prevStateLayer = this.layerContextWindow.getPrevStateLayer()

    this.setGates(prevStateLayer) // must be called before accessing to the activated values of the gates

    val y = this.outputArray.values
    val c = this.candidate.values
    val inG = this.inputGate.values
    val forG = this.forgetGate.values

    // y = inG * c
    y.assignProd(inG, c)

    // y += f(yPrev) * forG
    if (prevStateLayer != null) {
      val yPrev = prevStateLayer.outputArray.values
      this.activatedPrevOutput = if (this.activationFunction != null) this.activationFunction.f(yPrev) else yPrev

      y.assignSum(this.activatedPrevOutput!!.prod(forG))
    }
  }

  /**
   * Set gates values
   *
   * inG = sigmoid(wIn (dot) x + bIn + wrIn (dot) yPrev)
   * forG = sigmoid(wForG (dot) x + bForG + wrForG (dot) yPrev)
   * c = f(wc (dot) x)
   */
  private fun setGates(prevStateLayer: LayerStructure?): Unit { this.params as CFNLayerParameters

    val x = this.inputArray.values
    val c = this.candidate.values
    val wc = this.params.candidateWeights.values

    this.inputGate.forward(this.params.inputGate, x)
    this.forgetGate.forward(this.params.forgetGate, x)
    c.assignDot(wc, x)

    if (prevStateLayer != null) { // recurrent contribute for input and forget gates
      val yPrev = prevStateLayer.outputArray.values
      this.inputGate.addRecurrentContribute(this.params.inputGate, yPrev)
      this.forgetGate.addRecurrentContribute(this.params.forgetGate, yPrev)
    }

    this.inputGate.activate()
    this.forgetGate.activate()
    this.candidate.activate()
  }

  /**
   *
   */
  override fun backward(paramsErrors: LayerParameters, propagateToInput: Boolean) {

    this.paramsErrors = paramsErrors as CFNLayerParameters

    val prevStateLayer = this.layerContextWindow.getPrevStateLayer() as? CFNLayerStructure
    val nextStateLayer = this.layerContextWindow.getNextStateLayer() as? CFNLayerStructure

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
  private fun assignGatesGradients(prevStateLayer: CFNLayerStructure?): Unit { this.params as CFNLayerParameters

    val gy = this.outputArray.errors

    val inputGate = this.inputGate
    val forgetGate = this.forgetGate
    val candidate = this.candidate

    val ingG = inputGate.values
    val c = candidate.values

    val inGDeriv = inputGate.calculateActivationDeriv()
    val cDeriv = candidate.calculateActivationDeriv()

    val gInG = this.inputGate.errors
    val gc = this.candidate.errors

    gInG.assignProd(c, inGDeriv).assignProd(gy)
    gc.assignProd(ingG, cDeriv).assignProd(gy)

    if (prevStateLayer != null) {
      val aPrev = this.activatedPrevOutput!!
      val forGDeriv = forgetGate.calculateActivationDeriv()
      val gForG = this.forgetGate.errors

      gForG.assignProd(aPrev, forGDeriv).assignProd(gy)
    }
  }

  /**
   *
   * @param prevStateOutput the outputArray in the previous state
   */
  private fun assignParamsGradients(prevStateOutput: AugmentedArray?): Unit {

    val x = this.inputArray.values
    val yPrev = prevStateOutput?.values

    this.setGateParamsGradients(this.inputGate, this.paramsErrors!!.inputGate, x, yPrev = yPrev)
    this.setGateParamsGradients(this.forgetGate, this.paramsErrors!!.forgetGate, x, yPrev = yPrev)

    val gc = this.candidate.errors
    val gwc = this.paramsErrors!!.candidateWeights.values
    gwc.assignDot(gc, x.T)
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
                                     yPrev: NDArray? = null): Unit {

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
  private fun assignLayerGradients() { this.params as CFNLayerParameters

    val gx = this.inputArray.errors

    val wInG = this.params.inputGate.weights.values
    val wForG = this.params.forgetGate.weights.values
    val wC = this.params.candidateWeights.values

    val gInG = this.inputGate.errors
    val gForG = this.forgetGate.errors
    val gC = this.candidate.errors

    gx.assignValues(gForG.T.dot(wForG)).assignSum(gC.T.dot(wC)).assignSum(gInG.T.dot(wInG))

    if (this.inputArray.hasActivation) {
      gx.assignProd(this.inputArray.calculateActivationDeriv())
    }
  }

  /**
   *
   * @param nextStateLayer the layer structure in the next state
   */
  private fun addOutputRecurrentGradients(nextStateLayer: CFNLayerStructure?) {

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
  private fun getLayerRecurrentContribute(nextStateLayer: CFNLayerStructure): NDArray {
    this.params as CFNLayerParameters

    val inputGate = nextStateLayer.inputGate
    val forgetGate = nextStateLayer.forgetGate

    val gyNext = nextStateLayer.outputArray.errors

    val yDeriv = if (nextStateLayer.activationFunction != null)
      nextStateLayer.activationFunction.dfOptimized(nextStateLayer.activatedPrevOutput!!)
    else
      nextStateLayer.activatedPrevOutput!!

    val forG = forgetGate.values

    val gInG = inputGate.errors
    val gForG = forgetGate.errors

    val wrInG = this.params.inputGate.recurrentWeights.values
    val wrForG = this.params.forgetGate.recurrentWeights.values

    val gRec1 = forG.prod(yDeriv).assignProd(gyNext)
    val gRec2 = gInG.T.dot(wrInG)
    val gRec3 = gForG.T.dot(wrForG)

    return gRec1.assignSum(gRec2).assignSum(gRec3)
  }
}
