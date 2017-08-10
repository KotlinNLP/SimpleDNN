/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.birnn

/**
 * The accumulator of the errors of the attention network parameters.
 *
 * @property network the network whose to accumulate parameters
 */
class BiRNNParamsErrorsAccumulator(val network: BiRNN) {

  /**
   * A counter of times when errors were accumulated.
   */
  var count = 0
    private set

  /**
   * A boolean indicating whether errors are accumulated.
   */
  private val isEmpty: Boolean get() = this.count == 0

  /**
   * The structure in which to accumulate the errors of the BiRNN parameters.
   */
  private val paramsErrors: BiRNNParameters = this.paramsErrorsFactory()

  /**
   * @return the errors of the BiRNN parameters
   */
  fun getParamsErrors() = if (this.isEmpty) this.paramsErrorsFactory() else this.paramsErrors

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

    if (this.count != 0) {
      val countDouble = this.count.toDouble()
      this.paramsErrors.leftToRight.forEach { it.values.assignDiv(countDouble) }
      this.paramsErrors.rightToLeft.forEach { it.values.assignDiv(countDouble) }
    }
  }

  /**
   * Accumulate the given [paramsErrors] into the accumulator.
   *
   * @param paramsErrors the [BiRNNParameters] errors to accumulate
   */
  fun accumulate(paramsErrors: BiRNNParameters) {

    if (this.isEmpty) {
      this.assignValues(paramsErrors)
    } else {
      this.assignSum(paramsErrors)
    }

    this.count += 1
  }

  /**
   * @return the structure in which to accumulate the errors of the BiRNN parameters
   */
  private fun paramsErrorsFactory() = BiRNNParameters(
    leftToRight = this.network.leftToRightNetwork.parametersErrorsFactory(),
    rightToLeft = this.network.rightToLeftNetwork.parametersErrorsFactory()
  )

  /**
   * Assign the values of the given [paramsErrors] to the ones of the accumulator.
   *
   * @param paramsErrors the [BiRNNParameters] errors to assign
   */
  private fun assignValues(paramsErrors: BiRNNParameters) {

    this.paramsErrors.leftToRight.zip(paramsErrors.leftToRight).forEach { (a, b) ->
      a.values.assignValues(b.values)
    }

    this.paramsErrors.rightToLeft.zip(paramsErrors.rightToLeft).forEach { (a, b) ->
      a.values.assignValues(b.values)
    }
  }

  /**
   * Add the values of the given [paramsErrors] to the ones of the accumulator.
   *
   * @param paramsErrors the [BiRNNParameters] errors to add
   */
  private fun assignSum(paramsErrors: BiRNNParameters) {

    this.paramsErrors.leftToRight.zip(paramsErrors.leftToRight).forEach { (a, b) ->
      a.values.assignSum(b.values)
    }

    this.paramsErrors.rightToLeft.zip(paramsErrors.rightToLeft).forEach { (a, b) ->
      a.values.assignSum(b.values)
    }
  }
}
