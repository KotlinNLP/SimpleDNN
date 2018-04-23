/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.optimizer

/**
 *
 */
class ParamsErrorsAccumulator<ParamsErrorsType: IterableParams<ParamsErrorsType>> {

  /**
   * A counter of times when errors were accumulated.
   */
  var count = 0
    private set

  /**
   * A boolean indicating if any errors are accumulated.
   */
  val isEmpty: Boolean get() = this.count == 0

  /**
   * A boolean indicating if no errors were accumulated.
   */
  val isNotEmpty: Boolean get() = this.count > 0

  /**
   * The structure in which to accumulate the parameters errors.
   */
  private lateinit var paramsErrors: ParamsErrorsType

  /**
   * A boolean which indicates if [paramsErrors] is a reference of one given by the user or is created privately.
   */
  private var paramsErrorsByReference: Boolean = false

  /**
   * @param copy a Boolean indicating if the returned errors must be a copy or a reference
   *
   * @return the accumulated errors of the network parameters
   */
  fun getParamsErrors(copy: Boolean = true): ParamsErrorsType {
    require(this.isNotEmpty) { "Cannot get params errors without accumulating before" }

    return this.paramsErrors.let { if (copy) it.copy() else it }
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
   * @param paramsErrors the parameters errors to accumulate
   * @param copy a Boolean indicating if [paramsErrors] can be used as reference or must be copied. Set copy = false
   *             to optimize the accumulation when done only 1 time before the update. (default = true)
   */
  fun accumulate(paramsErrors: ParamsErrorsType, copy: Boolean = true) {

    if (this.isEmpty) {
      if (copy) {
        // Assignment
        this.assignParamsErrors(paramsErrors)
        this.paramsErrorsByReference = false

      } else {
        // Replacement (by reference)
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

  /**
   * Assign the given [paramsErrors] to the paramsErrors of this [ParamsErrorsAccumulator] (create a new one if still
   * not initialized).
   *
   * @param paramsErrors the parameters errors to assign to this [ParamsErrorsAccumulator]
   */
  private fun assignParamsErrors(paramsErrors: ParamsErrorsType) {

    try {
      this.paramsErrors.assignValues(paramsErrors)

    } catch (e: UninitializedPropertyAccessException) {
      this.paramsErrors = paramsErrors.copy()
    }
  }
}
