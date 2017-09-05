/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.optimizer

import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray

/**
 * The optimizer of the [NeuralNetwork] parameters.
 *
 * @param neuralNetwork the Attention Network to optimize
 * @param updateMethod the update method helper (Learning Rate, ADAM, AdaGrad, ...)
 */
class ParamsOptimizer(val neuralNetwork: NeuralNetwork, updateMethod: UpdateMethod) : Optimizer(updateMethod) {

  /**
   * The accumulator of errors of the network parameters.
   */
  private val paramsErrorsAccumulator: ParamsErrorsAccumulator = ParamsErrorsAccumulator(this.neuralNetwork)

  /**
   * Calculate the errors average, update the params.
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
   * @param copy a Boolean indicating if the [paramsErrors] can be used as reference or must be copied. Set copy = false
   *             to optimize the accumulation when the size of the examples batches is 1. (default = true)
   */
  fun accumulate(paramsErrors: NetworkParameters, copy: Boolean = true) {
    this.paramsErrorsAccumulator.accumulate(paramsErrors = paramsErrors, copy = copy)
  }

  /**
   * Update the params in respect of the errors using the update method helper (Learning Rate, ADAM, AdaGrad, ...).
   */
  private fun updateParams() {

    this.neuralNetwork.model.zip(this.paramsErrorsAccumulator.getParamsErrors(copy = false)).forEach {
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
