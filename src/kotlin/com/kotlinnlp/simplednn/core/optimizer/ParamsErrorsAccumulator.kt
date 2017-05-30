/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.optimizer

import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork

/**
 *
 * @param neuralNetwork
 */
class ParamsErrorsAccumulator(val neuralNetwork: NeuralNetwork) {

  /**
   *
   */
  var count = 0
    private set

  /**
   *
   */
  private val paramsErrors: NetworkParameters = this.neuralNetwork.parametersErrorsFactory()

  /**
   *
   */
  private val isEmpty: Boolean get() = this.count == 0

  /**
   *
   */
  fun getParamsErrors() = if (this.isEmpty) this.neuralNetwork.parametersErrorsFactory() else this.paramsErrors

  /**
   * Reset the avgLoss
   */
  fun reset() {
    this.count = 0
  }

  /**
   * Divide the accumulated gradients for the number of examples
   */
  fun averageErrors() {
    this.paramsErrors.assignDiv(this.count)
  }

  /**
   * Sum local errors to backwardParamsErrors
   *
   */
  fun accumulate(paramsErrors: NetworkParameters) {

    this.paramsErrors.let { if (this.isEmpty) it.assignValues(paramsErrors) else it.assignSum(paramsErrors) }

    this.count += 1
  }
}
