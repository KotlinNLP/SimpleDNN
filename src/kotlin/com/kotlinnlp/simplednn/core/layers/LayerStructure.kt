/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
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
abstract class LayerStructure<InputNDArrayType : NDArray<InputNDArrayType>>(
  val inputArray: AugmentedArray<InputNDArrayType>,
  val outputArray: AugmentedArray<DenseNDArray>,
  open val params: LayerParameters,
  val activationFunction: ActivationFunction? = null,
  val dropout: Double = 0.0) {

  /**
   * The probability to keep an output value (no dropout on it)
   */
  private val p = 1.0 - this.dropout

  /**
   * Set the errors of the inputArray
   * @param values the errors to set into the inputArray
   */
  fun setInput(values: InputNDArrayType) = this.inputArray.assignValues(values)

  /**
   * Set the errors of the outputArray
   * @param errors the errors to set into the outputArray
   */
  fun setErrors(errors: DenseNDArray) = this.outputArray.assignErrors(errors)

  /**
   * Forward the input to the output combining it with the parameters and apply dropout
   */
  fun forward(useDropout: Boolean = false) {

    if (useDropout) {
      this.applyDropout()
    }

    this.forwardInput()
  }

  /**
   * Forward the input to the output combining it with the parameters
   */
  abstract protected fun forwardInput()

  /**
   *
   */
  abstract fun backward(paramsErrors: LayerParameters, propagateToInput: Boolean = false)

  /**
   *
   */
  private fun applyDropout() {

    if (this.dropout > 0.0) {
      val inputShape = this.inputArray.values.shape
      val mask = this.inputArray.values.factory // mask of zeros and ones
        .random(inputShape)
        .roundInt(threshold = this.dropout)

      mask.assignDiv(this.p) // mask of zeros and [this.p]

      this.inputArray.values.assignProd(mask)
    }
  }
}
