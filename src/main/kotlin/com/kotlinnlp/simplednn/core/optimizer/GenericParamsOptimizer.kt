/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.optimizer

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray
import com.kotlinnlp.simplednn.utils.scheduling.BatchScheduling
import com.kotlinnlp.simplednn.utils.scheduling.EpochScheduling
import com.kotlinnlp.simplednn.utils.scheduling.ExampleScheduling

/**
 * The optimizer of neural parameters.
 *
 * @param updateMethod the update method helper (Learning Rate, ADAM, AdaGrad, ...)
 */
class GenericParamsOptimizer(private val updateMethod: UpdateMethod<*>) : ScheduledUpdater {

  /**
   * The accumulator of parameters errors.
   */
  private val paramsErrorsAccumulator = GenericParamsErrorsAccumulator()

  /**
   * Calculate the errors average, update the parameters.
   */
  override fun update() {

    if (this.paramsErrorsAccumulator.isNotEmpty) {
      
      this.paramsErrorsAccumulator.averageErrors()
      this.updateParams()
      this.paramsErrorsAccumulator.clear()
    }
  }

  /**
   * Accumulate the given [paramsErrors] into the accumulator.
   *
   * @param paramsErrors the parameters errors to accumulate
   * @param copy a Boolean indicating if the [paramsErrors] can be used as reference or must be copied. Set copy = false
   *             to optimize the accumulation when the amount of the errors to accumulate is 1. (default = true)
   */
  fun accumulate(paramsErrors: ParamsArray.Errors<*>, copy: Boolean = true) {

    this.paramsErrorsAccumulator.accumulate(paramsErrors = paramsErrors, copy = copy)
  }

  /**
   * Method to call every new epoch.
   * In turn it calls the same method into the `updateMethod`
   */
  override fun newEpoch() {

    if (this.updateMethod is EpochScheduling) {
      this.updateMethod.newEpoch()
    }
  }

  /**
   * Method to call every new batch.
   * In turn it calls the same method into the `updateMethod`
   */
  override fun newBatch() {

    if (this.updateMethod is BatchScheduling) {
      this.updateMethod.newBatch()
    }
  }

  /**
   * Method to call every new example.
   * In turn it calls the same method into the `updateMethod`
   */
  override fun newExample() {

    if (this.updateMethod is ExampleScheduling) {
      this.updateMethod.newExample()
    }
  }

  /**
   * Update the parameters in respect of the errors using the update method helper (Learning Rate, ADAM, AdaGrad, ...).
   */
  private fun updateParams() {

    this.paramsErrorsAccumulator.getParamsErrors(copy = false).forEach {

      val params = it.refParams
      val errors = it.values

      when (errors) {
        is DenseNDArray -> this.updateMethod.update(array = params, errors = errors)
        is SparseNDArray -> this.updateMethod.update(array = params, errors = errors)
        else -> throw RuntimeException("Invalid errors type")
      }
    }
  }
}
