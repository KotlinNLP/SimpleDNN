/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attentionnetwork

import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.optimizer.Optimizer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray

/**
 * The optimizer of the parameters of the [AttentionNetwork]
 *
 * @param model the [AttentionNetworkParameters] to optimize
 * @param updateMethod the update method helper (Learning Rate, ADAM, AdaGrad, ...)
 */
class AttentionNetworkOptimizer(
  val model: AttentionNetworkParameters,
  updateMethod: UpdateMethod
) : Optimizer(updateMethod) {

  /**
   * The accumulator of errors of the network parameters.
   */
  private val paramsErrorsAccumulator = AttentionNetworkParamsErrorsAccumulator(
    inputSize = this.model.inputSize,
    attentionSize = this.model.attentionSize,
    sparseInput = this.model.sparseInput
  )

  /**
   * Calculate the errors average and update the params.
   */
  override fun update() {

    this.paramsErrorsAccumulator.averageErrors()
    this.updateParams()
    this.paramsErrorsAccumulator.reset()
  }

  /**
   * Accumulate the given [paramsErrors] into the accumulator.
   *
   * @param paramsErrors the network parameters errors to accumulate
   */
  fun accumulate(paramsErrors: AttentionNetworkParameters) {
    this.paramsErrorsAccumulator.accumulate(paramsErrors)
  }

  /**
   * Update the params in respect of the errors using the update method helper (Learning Rate, ADAM, AdaGrad, ...).
   */
  private fun updateParams() {

    val accumulatedErrors: AttentionNetworkParameters = this.paramsErrorsAccumulator.getParamsErrors()

    this.updateMethod.update(
      array = this.model.attentionParams.contextVector,
      errors = accumulatedErrors.attentionParams.contextVector.values)

    this.model.transformParams.zip(accumulatedErrors.transformParams).forEach { (params, errors) ->

      val e = errors.values

      when (e) {
        is DenseNDArray -> this.updateMethod.update(array = params as UpdatableDenseArray, errors = e)
        is SparseNDArray -> this.updateMethod.update(array = params as UpdatableDenseArray, errors = e)
        else -> throw RuntimeException("Invalid errors type")
      }
    }
  }
}
