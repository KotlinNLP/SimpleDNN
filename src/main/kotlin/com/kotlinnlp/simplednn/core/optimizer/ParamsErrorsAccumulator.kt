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
  private var paramsErrors: NetworkParameters = this.neuralNetwork.parametersErrorsFactory()

  /**
   * A boolean which indicates if [paramsErrors] is a reference of one given by the user or is created privately.
   */
  private var paramsErrorsByReference: Boolean = false

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

    if (this.count > 1) {
      this.paramsErrors.assignDiv(this.count)
    }
  }

  /**
   * Accumulate the given [paramsErrors] into the accumulator.
   *
   * @param paramsErrors the network parameters errors to accumulate
   * @param copy a Boolean indicating if the [paramsErrors] can be used as reference or must be copied. Set copy = false
   *             to optimize the accumulation when the size of the examples batches is 1. (default = true)
   */
  fun accumulate(paramsErrors: NetworkParameters, copy: Boolean = true) {

    if (this.isEmpty) {
      // Assignment
      if (copy) {
        this.paramsErrors.assignValues(paramsErrors)
        this.paramsErrorsByReference = false
      } else {
        this.paramsErrors = paramsErrors
        this.paramsErrorsByReference = true
      }

    } else {
      // Summation
      require(!this.paramsErrorsByReference) { "Cannot accumulate errors into paramsErrors given by reference" }
      this.paramsErrors.assignSum(paramsErrors)
    }

    this.count += 1
  }
}
