/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.optimizer

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray

/**
 * Generic params errors accumulator.
 */
open class ParamsErrorsAccumulator {

  /**
   * Handle the accumulation of a single ParamsErrors.
   *
   * @property paramsErrors the accumulated params errors
   * @param byReference whether the [paramsErrors] are by reference or a copy
   */
  class AccumulatedParamsErrors private constructor(
    val paramsErrors: ParamsArray.Errors<*>,
    private val byReference: Boolean
  ) {

    companion object {

      /**
       * @param paramsErrors the params errors to accumulate
       *
       * @return a new [AccumulatedParamsErrors] where the [paramsErrors] are copy
       */
      fun byCopy(paramsErrors: ParamsArray.Errors<*>) =
        AccumulatedParamsErrors(paramsErrors.copy(), byReference = false)

      /**
       * @param paramsErrors the params errors to accumulate
       *
       * @return a new [AccumulatedParamsErrors] where the [paramsErrors] are by reference
       */
      fun byReference(paramsErrors: ParamsArray.Errors<*>) =
        AccumulatedParamsErrors(paramsErrors, byReference = true)
    }

    /**
     * Count the number of accumulation.
     */
    private var count: Int = 1

    /**
     * Sum the [other] params errors to the [paramsErrors].
     *
     * @param other the params errors to accumulate
     *
     * @throws IllegalArgumentException if the [paramsErrors] are by reference
     */
    fun accumulate(other: ParamsArray.Errors<*>) {

      require(!this.byReference) { "Cannot accumulate errors into paramsErrors given by reference" }

      this.paramsErrors.values.assignSum(other.values)
      this.count++
    }

    /**
     * Average the accumulated the [paramsErrors] dividing each parameter by [count].
     */
    fun averageErrors() {

      if (this.count > 1) {

        this.paramsErrors.values.assignDiv(this.count.toDouble())

        this.count = 1
      }
    }
  }

  /**
   * A boolean indicating if any errors are accumulated.
   */
  val isEmpty: Boolean get() = this.paramsErrorsMap.isEmpty()

  /**
   * A boolean indicating if no errors were accumulated.
   */
  val isNotEmpty: Boolean get() = this.paramsErrorsMap.isNotEmpty()

  /**
   * The structure in which to accumulate the parameters errors.
   */
  private val paramsErrorsMap = mutableMapOf<String, AccumulatedParamsErrors>()

  /**
   * Accumulate the given [paramsErrors] into the accumulator.
   *
   * @param paramsErrors the parameters errors to accumulate
   * @param copy a Boolean indicating if [paramsErrors] can be used as reference or must be copied. Set copy = false
   *             to optimize the accumulation when immediately followed by an update. (default = true)
   */
  fun accumulate(paramsErrors: ParamsArray.Errors<*>, copy: Boolean = true) {

    val paramsUUID = paramsErrors.refParams.uuid

    when {
      paramsUUID in this.paramsErrorsMap -> this.paramsErrorsMap.getValue(paramsUUID).accumulate(paramsErrors)
      copy -> this.paramsErrorsMap[paramsUUID] = AccumulatedParamsErrors.byCopy(paramsErrors)
      else -> this.paramsErrorsMap[paramsUUID] = AccumulatedParamsErrors.byReference(paramsErrors)
    }
  }

  /**
   * Accumulate the given list of [paramsErrors] into the accumulator.
   *
   * @param paramsErrors the parameters errors to accumulate
   * @param copy a Boolean indicating if [paramsErrors] can be used as reference or must be copied. Set copy = false
   *             to optimize the accumulation when immediately followed by an update. (default = true)
   */
  fun accumulate(paramsErrors: ParamsErrorsList, copy: Boolean = true) {

    paramsErrors.map { this.accumulate(it, copy = copy) }
  }

  /**
   * Accumulate the given params [errors] into the accumulator.
   *
   * @param params the parameters
   * @param errors the errors of the given [params] to accumulate
   */
  fun accumulate(params: ParamsArray, errors: DenseNDArray) = this.accumulate(params.buildDenseErrors(errors))

  /**
   * Accumulate the given params [errors] into the accumulator.
   *
   * @param params the parameters
   * @param errors the errors of the given [params] to accumulate
   */
  fun accumulate(params: ParamsArray, errors: SparseNDArray) = this.accumulate(params.buildSparseErrors(errors))

  /**
   * Divide the accumulated errors by the number of accumulations.
   */
  fun averageErrors() = this.paramsErrorsMap.values.forEach { it.averageErrors() }

  /**
   * @param copy a Boolean indicating if the returned errors must be a copy or a reference
   *
   * @return the accumulated errors of the network parameters
   */
  fun getParamsErrors(copy: Boolean = true): ParamsErrorsList =
    this.paramsErrorsMap.values.map { if (copy) it.paramsErrors.copy() else it.paramsErrors }

  /**
   * Clear the accumulated errors.
   */
  fun clear() = this.paramsErrorsMap.clear()
}
