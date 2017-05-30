/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.feedforward

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerStructure
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * @param inputArray the input array of the layer
 * @param outputArray the output array of the layer
 * @param params the parameters which connect the input to the output
 * @param activationFunction the activation function of the layer
 * @param dropout The probability of dropout (default 0.0).
 *                If applying it, the usual value is 0.5 (better 0.25 if it's the first layer).
 */
class FeedforwardLayerStructure<InputNDArrayType : NDArray<InputNDArrayType>>(
  inputArray: AugmentedArray<InputNDArrayType>,
  outputArray: AugmentedArray<DenseNDArray>,
  params: LayerParameters,
  activationFunction: ActivationFunction? = null,
  dropout: Double = 0.0
) : LayerStructure<InputNDArrayType>(
  inputArray = inputArray,
  outputArray = outputArray,
  params = params,
  activationFunction = activationFunction,
  dropout = dropout) {

  /**
   * A support variable to manage the errors on the parameters during the backward
   */
  var paramsErrors: FeedforwardLayerParameters? = null

  /**
   * Initialization: set the activation function to the outputArray
   */
  init {
    if (activationFunction != null) {
      outputArray.setActivation(activationFunction)
    }
  }

  /**
   * Forward the input to the output combining it with the parameters
   *
   * y = f(w (dot) x + b)
   */
  override fun forwardInput() { this.params as FeedforwardLayerParameters

    val x: InputNDArrayType = this.inputArray.values
    val y: DenseNDArray = this.outputArray.values

    val w: DenseNDArray = this.params.weights.values as DenseNDArray
    val b: DenseNDArray = this.params.biases.values

    y.assignDot(w, x).assignSum(b)

    this.outputArray.activate()
  }

  /**
   *
   */
  override fun backward(paramsErrors: LayerParameters, propagateToInput: Boolean) {

    this.paramsErrors = paramsErrors as FeedforwardLayerParameters

    this.assignParamsGradients()

    if (propagateToInput) {
      this.assignLayerGradients()
    }
  }

  /**
   * gb = gy * 1
   * gw = gy (dot) x
   */
  private fun assignParamsGradients() {

    val gb: DenseNDArray = this.paramsErrors!!.biases.values
    val gw: NDArray<*> = this.paramsErrors!!.weights.values

    val x: InputNDArrayType = this.inputArray.values
    val gy: DenseNDArray = this.outputArray.errors

    gb.assignValues(gy)
    gw.assignDot(gy, x.T)
  }

  /**
   * gx = (gy (dot) w) * xDeriv
   */
  private fun assignLayerGradients() { this.params as FeedforwardLayerParameters

    val gy: DenseNDArray = this.outputArray.errors
    val w: DenseNDArray = this.params.weights.values as DenseNDArray

    val gx: DenseNDArray = this.inputArray.errors

    gx.assignValues(gy.T.dot(w))

    if (this.inputArray.hasActivation && gx is DenseNDArray) {
      gx.assignProd(this.inputArray.calculateActivationDeriv())
    }
  }
}
