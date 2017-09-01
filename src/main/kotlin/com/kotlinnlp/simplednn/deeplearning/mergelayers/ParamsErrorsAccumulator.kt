/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.mergelayers

/**
 * The accumulator of parameters errors of a [MergeLayer].
 *
 * @property layer
 */
class ParamsErrorsAccumulator(val layer: MergeLayer<*>) {

  /**
   * A counter of times when errors were accumulated.
   */
  var count = 0
    private set

  /**
   * The structure in which to accumulate the errors of the merge layer parameters.
   */
  private val paramsErrors: MergeLayerParameters = this.layer.parametersErrorsFactory()

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
  fun getParamsErrors(copy: Boolean = true): MergeLayerParameters {
    require(this.isNotEmpty) { "Cannot get params errors without accumulating before" }

    val paramsErrors: MergeLayerParameters

    return if (copy) {

      paramsErrors = this.layer.parametersErrorsFactory()

      paramsErrors.zip(this.paramsErrors).forEach { (emptyErrors, errors) ->
        emptyErrors.values.assignValues(errors.values)
      }

      paramsErrors

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
    this.paramsErrors.forEach { it.values.assignDiv(this.count.toDouble()) }
  }

  /**
   * Accumulate the given [paramsErrors] into the accumulator.
   *
   * @param paramsErrors the merge layer parameters errors to accumulate
   */
  fun accumulate(paramsErrors: MergeLayerParameters) {

    if (this.isEmpty) {
      this.paramsErrors.zip(paramsErrors).forEach { (errors, newErrors) ->
        errors.values.assignValues(newErrors.values)
      }

    } else {
      this.paramsErrors.zip(paramsErrors).forEach { (errors, newErrors) ->
        errors.values.assignSum(newErrors.values)
      }
    }

    this.count += 1
  }
}
