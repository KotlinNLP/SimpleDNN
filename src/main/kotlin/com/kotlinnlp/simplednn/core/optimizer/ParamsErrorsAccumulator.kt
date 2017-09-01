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
   * A counter of times when errors were accumulated.
   */
  var count = 0
    private set

  /**
   * The structure in which to accumulate the errors of the network parameters.
   */
  private val paramsErrors: NetworkParameters = this.neuralNetwork.parametersErrorsFactory()

  /**
   * A boolean indicating if any errors are accumulated.
   */
  private val isEmpty: Boolean get() = this.count == 0

  /**
   * A boolean indicating if no errors were accumulated.
   */
  private val isNotEmpty: Boolean get() = this.count > 0

  /**
   * @param copy a Boolean indicating if the returned errors must be a copy or a reference
   *
   * @return the accumulated errors of the network parameters
   */
  fun getParamsErrors(copy: Boolean = true): NetworkParameters {
    require(this.isNotEmpty) { "Cannot get params errors without accumulating before" }

    val paramsError: NetworkParameters

    return if (copy) {
      paramsError = this.neuralNetwork.parametersErrorsFactory()
      paramsError.assignValues(this.paramsErrors)
      paramsError

    } else {
      this.paramsErrors
    }
  }

  /**
   * Reset the accumulated errors.
   */
  fun reset() {
    this.count = 0
  }

  /**
   * Divide the accumulated errors by the number of accumulations.
   */
  fun averageErrors() {
    this.paramsErrors.assignDiv(this.count)
  }

  /**
   * Accumulate the given [paramsErrors] into the accumulator.
   *
   * @param paramsErrors the network parameters errors to accumulate
   */
  fun accumulate(paramsErrors: NetworkParameters) {

    this.paramsErrors.let { if (this.isEmpty) it.assignValues(paramsErrors) else it.assignSum(paramsErrors) }

    this.count += 1
  }
}
