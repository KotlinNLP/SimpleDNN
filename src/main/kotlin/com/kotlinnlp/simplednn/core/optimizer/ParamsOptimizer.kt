/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.optimizer

import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray

/**
 * The optimizer of neural parameters.
 *
 * @param params the parameters to optimize
 * @param updateMethod the update method helper (Learning Rate, ADAM, AdaGrad, ...)
 */
class ParamsOptimizer<ParamsType: IterableParams<ParamsType>>(
  val params: ParamsType,
  updateMethod: UpdateMethod
) : Optimizer(updateMethod) {

  /**
   * The accumulator of parameters errors.
   */
  private val paramsErrorsAccumulator = ParamsErrorsAccumulator<ParamsType>()

  /**
   * A Boolean indicating if errors have been accumulated.
   */
  private var isEmpty: Boolean = false

  /**
   * A Boolean indicating if no errors have been accumulated.
   */
  private val isNotEmpty: Boolean get() = !this.isEmpty

  /**
   * Calculate the errors average, update the parameters.
   */
  override fun update() {

    if (this.isNotEmpty) {
      
      this.paramsErrorsAccumulator.averageErrors()
      this.updateParams()
      this.paramsErrorsAccumulator.reset()

      this.isEmpty = true
    }
  }

  /**
   * Accumulate the given [paramsErrors] into the accumulator.
   *
   * @param paramsErrors the parameters errors to accumulate
   * @param copy a Boolean indicating if the [paramsErrors] can be used as reference or must be copied. Set copy = false
   *             to optimize the accumulation when the amount of the errors to accumulate is 1. (default = true)
   */
  fun accumulate(paramsErrors: ParamsType, copy: Boolean = true) {

    require(this.params.zip(paramsErrors).all{ (params, errors) -> params.values.shape == errors.values.shape }) {
      "paramsErrors contains arrays with not compatible shape"
    }

    this.paramsErrorsAccumulator.accumulate(paramsErrors = paramsErrors, copy = copy)

    this.isEmpty = false
  }

  /**
   * Update the parameters in respect of the errors using the update method helper (Learning Rate, ADAM, AdaGrad, ...).
   */
  private fun updateParams() {

    this.params.zip(this.paramsErrorsAccumulator.getParamsErrors(copy = false)).forEach {
      (params, errors) ->

      val e = errors.values

      when (e) {
        is DenseNDArray -> this.updateMethod.update(array = params as UpdatableDenseArray, errors = e)
        is SparseNDArray -> this.updateMethod.update(array = params as UpdatableDenseArray, errors = e)
        else -> throw RuntimeException("Invalid errors type")
      }
    }
  }
}
