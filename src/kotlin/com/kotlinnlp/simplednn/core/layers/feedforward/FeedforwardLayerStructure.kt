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
import com.kotlinnlp.simplednn.simplemath.NDArray

/**
 * @param inputArray the input array of the layer
 * @param outputArray the output array of the layer
 * @param params the parameters which connect the input to the output
 * @param activationFunction the activation function of the layer
 * @param dropout The probability of dropout (default 0.0).
 *                If applying it, the usual value is 0.5 (better 0.25 if it's the first layer).
 */
class FeedforwardLayerStructure(
  inputArray: AugmentedArray,
  outputArray: AugmentedArray,
  params: LayerParameters,
  activationFunction: ActivationFunction? = null,
  dropout: Double = 0.0
) : LayerStructure(
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
  override fun forwardInput(): Unit { this.params as FeedforwardLayerParameters

    val x: NDArray = this.inputArray.values
    val y: NDArray = this.outputArray.values

    val w: NDArray = this.params.weights.values
    val b: NDArray = this.params.biases.values

    y.assignDot(w, x).assignSum(b)

    this.outputArray.activate()
  }

  /**
   *
   */
  override fun backward(paramsErrors: LayerParameters, propagateToInput: Boolean): Unit {

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
  private fun assignParamsGradients(): Unit {

    val gb = this.paramsErrors!!.biases.values
    val gw = this.paramsErrors!!.weights.values

    val x = this.inputArray.values
    val gy = this.outputArray.errors

    gb.assignValues(gy)
    gw.assignDot(gy, x.T)
  }

  /**
   * gx = (gy (dot) w) * xDeriv
   */
  private fun assignLayerGradients(): Unit { this.params as FeedforwardLayerParameters

    val gy = this.outputArray.errors
    val w = this.params.weights.values

    val gx = this.inputArray.errors

    gx.assignValues(gy.T.dot(w))

    if (this.inputArray.hasActivation) {
      gx.assignProd(this.inputArray.calculateActivationDeriv())
    }
  }
}
