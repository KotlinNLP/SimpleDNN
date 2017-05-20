/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.simple

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
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
class SimpleRecurrentLayerStructure(
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
  var paramsErrors: SimpleRecurrentLayerParameters? = null

  /**
   * Initialization: set the activation function to the outputArray
   */
  init {
    if (activationFunction != null) {
      outputArray.setActivation(activationFunction)
    }
  }

  /**
   * y = f(w (dot) x + b + wRec (dot) yPrev)
   */
  override fun forwardInput() { this.params as SimpleRecurrentLayerParameters

    val w: NDArray = this.params.weights.values
    val b: NDArray = this.params.biases.values

    val x: NDArray = this.inputArray.values
    val y: NDArray = this.outputArray.values

    // y = w (dot) x + b
    y.assignDot(w, x).assignSum(b)

    // y += wRec (dot) yAPrev
    val prevStateLayer = this.layerContextWindow.getPrevStateLayer()
    if (prevStateLayer != null) { // recurrent contribute

      val wRec: NDArray = this.params.recurrentWeights.values
      val yPrev: NDArray = prevStateLayer.outputArray.values

      y.assignSum(wRec.dot(yPrev))
    }

    this.outputArray.activate()
  }

  /**
   *
   */
  override fun backward(paramsErrors: LayerParameters, propagateToInput: Boolean) {

    this.paramsErrors = paramsErrors as SimpleRecurrentLayerParameters

    val nextStateLayer = this.layerContextWindow.getNextStateLayer()

    if (nextStateLayer != null) {
      this.addLayerRecurrentGradients(nextStateLayer)
    }

    this.assignParamsGradients(nextStateLayer)

    if (propagateToInput) {
      this.assignLayerGradients()
    }
  }

  /**
   * gb = gy * 1
   * gw = gb (dot) x
   * gwRec = gyNext (dot) y
   */
  private fun assignParamsGradients(nextStateLayer: LayerStructure?) {

    val x = this.inputArray.values
    val gb = this.paramsErrors!!.biases.values
    val gw = this.paramsErrors!!.weights.values
    val gy = this.outputArray.errors

    gb.assignValues(gy)
    gw.assignDot(gb, x.T)

    if (nextStateLayer != null) { // recurrent errors
      val gyNext = nextStateLayer.outputArray.errors
      val y: NDArray = this.outputArray.values
      val gwRec: NDArray = this.paramsErrors!!.recurrentWeights.values

      gwRec.assignDot(gyNext, y.T)
    }
  }

  /**
   * gx = (gb (dot) w) * xDeriv
   */
  private fun assignLayerGradients() { this.params as SimpleRecurrentLayerParameters

    val gb = this.paramsErrors!!.biases.values
    val w = this.params.weights.values

    val gx = this.inputArray.errors

    // gx = gb (dot) w
    gx.assignValues(gb.T.dot(w))

    // gx *= xDeriv
    if (this.inputArray.hasActivation) {
      val xDeriv = this.inputArray.calculateActivationDeriv()
      gx.assignProd(xDeriv)
    }
  }

  /**
   * gy += (gyNext (dot) wRec) * yDeriv
   */
  private fun addLayerRecurrentGradients(nextStateLayer: LayerStructure) {
    this.params as SimpleRecurrentLayerParameters

    val gy = this.outputArray.errors
    val gyNext: NDArray = nextStateLayer.outputArray.errors
    val wRec: NDArray = this.params.recurrentWeights.values

    // gRec = gyNext (dot) wRec
    val gRec = gyNext.T.dot(wRec)

    // gRec *= yDeriv
    if (this.outputArray.hasActivation) {
      val yDeriv = this.outputArray.calculateActivationDeriv()
      gRec.assignProd(yDeriv)
    }

    gy.assignSum(gRec)
  }
}
