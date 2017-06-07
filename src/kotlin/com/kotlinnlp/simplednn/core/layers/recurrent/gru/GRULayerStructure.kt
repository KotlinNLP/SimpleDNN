/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.gru

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.arrays.DistributionArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.activations.Sigmoid
import com.kotlinnlp.simplednn.core.layers.*
import com.kotlinnlp.simplednn.core.layers.recurrent.*
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * The CRU Layer Structure.
 *
 * @property inputArray the input array of the layer
 * @property outputArray the output array of the layer
 * @property params the parameters which connect the input to the output
 * @property layerContextWindow the context window used for the forward and the backward
 * @property activationFunction the activation function of the layer
 * @property dropout the probability of dropout (default 0.0).
 *                   If applying it, the usual value is 0.5 (better 0.25 if it's the first layer).
 */
class GRULayerStructure<InputNDArrayType : NDArray<InputNDArrayType>>(
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
  var paramsErrors: GRULayerParameters? = null

  /**
   *
   */
  val candidate = GateUnit<InputNDArrayType>(outputArray.size)

  /**
   *
   */
  val resetGate = GateUnit<InputNDArrayType>(outputArray.size)

  /**
   *
   */
  val partitionGate = GateUnit<InputNDArrayType>(outputArray.size)

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

    val y: DenseNDArray = this.outputArray.values
    val c: DenseNDArray = this.candidate.values
    val p: DenseNDArray = this.partitionGate.values

    // y = p * c
    y.assignProd(p, c)

    // y += (1 - p) * yPrev
    if (prevStateLayer != null) {
      val yPrev: DenseNDArray = prevStateLayer.outputArray.values
      y.assignSum(p.reverseSub(1.0).prod(yPrev))
    }
  }

  /**
   * Forward the input to the output combining it with the parameters, calculating its relevance respect of the output.
   *
   * @param relevantOutcomesDistribution the distribution which indicates which outcomes are relevant, used
   *                                     as reference to calculate the relevance of the input
   */
  override fun forwardInput(relevantOutcomesDistribution: DistributionArray) {
    TODO("not implemented")
  }


  /**
   * Set gates values
   *
   * r = sigmoid(wr (dot) x + br + wrRec (dot) yPrev)
   * p = sigmoid(wp (dot) x + bp + wpRec (dot) yPrev)
   * c = f(wc (dot) x + bc + wcRec (dot) (yPrev * r))
   */
  private fun setGates(prevStateLayer: LayerStructure<*>?) { this.params as GRULayerParameters

    val x: InputNDArrayType = this.inputArray.values

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

    this.addOutputRecurrentGradients(nextStateLayer as? GRULayerStructure<*>)

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
  private fun assignGatesGradients(prevStateOutput: AugmentedArray<DenseNDArray>?) {
    this.params as GRULayerParameters

    val gy: DenseNDArray = this.outputArray.errors

    val resetGate = this.resetGate
    val partitionGate = this.partitionGate
    val candidate = this.candidate

    val p: DenseNDArray = partitionGate.values
    val c: DenseNDArray = candidate.values

    val rDeriv: DenseNDArray = resetGate.calculateActivationDeriv()
    val pDeriv: DenseNDArray = partitionGate.calculateActivationDeriv()
    val cDeriv: DenseNDArray = candidate.calculateActivationDeriv()

    val gr: DenseNDArray = this.resetGate.errors
    val gp: DenseNDArray = this.partitionGate.errors
    val gc: DenseNDArray = this.candidate.errors

    gc.assignProd(p, cDeriv).assignProd(gy)  // gc must be calculated before gr and gp

    if (prevStateOutput == null) {
      gr.zeros()
      gp.assignProd(c, pDeriv).assignProd(gy)

    } else { // recurrent contribute

      val yPrev: DenseNDArray = prevStateOutput.values
      val wcr: DenseNDArray = this.params.candidate.recurrentWeights.values

      gr.assignValues(gc.T.dot(wcr)).assignProd(rDeriv).assignProd(yPrev)
      gp.assignProd(c.sub(yPrev), pDeriv).assignProd(gy)
    }
  }

  /**
   *
   * @param prevStateOutput the outputArray in the previous state
   */
  private fun assignParamsGradients(prevStateOutput: AugmentedArray<DenseNDArray>?) {

    val x: InputNDArrayType = this.inputArray.values
    val yPrev: DenseNDArray? = prevStateOutput?.values

    this.resetGate.assignParamsGradients(paramsErrors = this.paramsErrors!!.resetGate, x = x, yPrev = yPrev)
    this.partitionGate.assignParamsGradients(paramsErrors = this.paramsErrors!!.partitionGate, x = x, yPrev = yPrev)
    this.candidate.assignParamsGradients(paramsErrors = this.paramsErrors!!.candidate, x = x)

    if (yPrev != null) { // add recurrent contribute to the recurrent weights of the candidate
      val r: DenseNDArray = this.resetGate.values
      val gwcr: DenseNDArray = this.paramsErrors!!.candidate.recurrentWeights.values
      val gc: DenseNDArray = this.candidate.errors
      gwcr.assignDot(gc, r.prod(yPrev).T)
    }
  }

  /**
   *
   */
  private fun assignLayerGradients() { this.params as GRULayerParameters

    val gx: DenseNDArray = this.inputArray.errors

    val wp: DenseNDArray = this.params.partitionGate.weights.values as DenseNDArray
    val wc: DenseNDArray = this.params.candidate.weights.values as DenseNDArray
    val wr: DenseNDArray = this.params.resetGate.weights.values as DenseNDArray

    val gp: DenseNDArray = this.partitionGate.errors
    val gc: DenseNDArray = this.candidate.errors
    val gr: DenseNDArray = this.resetGate.errors

    gx.assignValues(gp.T.dot(wp)).assignSum(gc.T.dot(wc)).assignSum(gr.T.dot(wr))

    if (this.inputArray.hasActivation && gx is DenseNDArray) {
      gx.assignProd(this.inputArray.calculateActivationDeriv())
    }
  }

  /**
   *
   * @param nextStateLayer the layer structure in the next state
   */
  private fun addOutputRecurrentGradients(nextStateLayer: GRULayerStructure<*>?) {

    if (nextStateLayer != null) {
      val gy: DenseNDArray = this.outputArray.errors
      val gyRec: DenseNDArray = this.getLayerRecurrentContribute(nextStateLayer)

      gy.assignSum(gyRec)
    }
  }

  /**
   *
   * @param nextStateLayer the layer structure in the next state
   */
  private fun getLayerRecurrentContribute(nextStateLayer: GRULayerStructure<*>): DenseNDArray {
    this.params as GRULayerParameters

    val resetGate = nextStateLayer.resetGate
    val partitionGate = nextStateLayer.partitionGate
    val candidate = nextStateLayer.candidate

    val gy: DenseNDArray = nextStateLayer.outputArray.errors

    val r: DenseNDArray = resetGate.values
    val p: DenseNDArray = partitionGate.values

    val gr: DenseNDArray = resetGate.errors
    val gp: DenseNDArray = partitionGate.errors
    val gc: DenseNDArray = candidate.errors

    val wrr: DenseNDArray = this.params.resetGate.recurrentWeights.values
    val wpr: DenseNDArray = this.params.partitionGate.recurrentWeights.values
    val wcr: DenseNDArray = this.params.candidate.recurrentWeights.values

    val gRec1: DenseNDArray = gr.T.dot(wrr)
    val gRec2: DenseNDArray = gp.T.dot(wpr)
    val gRec3: DenseNDArray = gc.T.dot(wcr).prod(r)
    val gRec4: DenseNDArray = p.reverseSub(1.0).prod(gy).T

    return gRec1.assignSum(gRec2).assignSum(gRec3).assignSum(gRec4)
  }
}
