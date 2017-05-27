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
class SimpleRecurrentLayerStructure<InputNDArrayType : NDArray<InputNDArrayType>>(
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

    val w: DenseNDArray = this.params.weights.values as DenseNDArray
    val b: DenseNDArray = this.params.biases.values

    val x: NDArray<*> = this.inputArray.values
    val y: DenseNDArray = this.outputArray.values

    // y = w (dot) x + b
    y.assignDot(w, x).assignSum(b)

    // y += wRec (dot) yAPrev
    val prevStateLayer = this.layerContextWindow.getPrevStateLayer()
    if (prevStateLayer != null) { // recurrent contribute

      val wRec: DenseNDArray = this.params.recurrentWeights.values
      val yPrev: DenseNDArray = prevStateLayer.outputArray.values

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
  private fun assignParamsGradients(nextStateLayer: LayerStructure<*>?) {

    val x: InputNDArrayType = this.inputArray.values
    val gb: DenseNDArray = this.paramsErrors!!.biases.values
    val gw: DenseNDArray = this.paramsErrors!!.weights.values as DenseNDArray
    val gy: DenseNDArray = this.outputArray.errors

    gb.assignValues(gy)
    gw.assignDot(gb, x.T)

    if (nextStateLayer != null) { // recurrent errors
      val gyNext: DenseNDArray = nextStateLayer.outputArray.errors
      val y: DenseNDArray = this.outputArray.values
      val gwRec: DenseNDArray = this.paramsErrors!!.recurrentWeights.values

      gwRec.assignDot(gyNext, y.T)
    }
  }

  /**
   * gx = (gb (dot) w) * xDeriv
   */
  private fun assignLayerGradients() { this.params as SimpleRecurrentLayerParameters

    val gb: DenseNDArray = this.paramsErrors!!.biases.values
    val w: DenseNDArray = this.params.weights.values as DenseNDArray

    val gx: DenseNDArray = this.inputArray.errors

    // gx = gb (dot) w
    gx.assignValues(gb.T.dot(w))

    // gx *= xDeriv
    if (this.inputArray.hasActivation && gx is DenseNDArray) {
      gx.assignProd(this.inputArray.calculateActivationDeriv())
    }
  }

  /**
   * gy += (gyNext (dot) wRec) * yDeriv
   */
  private fun addLayerRecurrentGradients(nextStateLayer: LayerStructure<*>) {
    this.params as SimpleRecurrentLayerParameters

    val gy: DenseNDArray = this.outputArray.errors
    val gyNext: DenseNDArray = nextStateLayer.outputArray.errors
    val wRec: DenseNDArray = this.params.recurrentWeights.values

    // gRec: com.kotlinnlp.simplednn.simplemath.ndarray.DenseNDArray = gyNext (dot) wRec
    val gRec: DenseNDArray = gyNext.T.dot(wRec)

    // gRec *= yDeriv
    if (this.outputArray.hasActivation) {
      val yDeriv: DenseNDArray = this.outputArray.calculateActivationDeriv()
      gRec.assignProd(yDeriv)
    }

    gy.assignSum(gRec)
  }
}
