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
import com.kotlinnlp.simplednn.simplemath.ndarray.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.SparseNDArray
import com.kotlinnlp.simplednn.utils.scheduling.BatchScheduling
import com.kotlinnlp.simplednn.utils.scheduling.EpochScheduling
import com.kotlinnlp.simplednn.utils.scheduling.ExampleScheduling

/**
 *
 * @param neuralNetwork neural layer
 */
class ParamsOptimizer(
  val neuralNetwork: NeuralNetwork,
  val updateMethod: UpdateMethod,
  val minLossCountToUpdate: Int = 1
) : Optimizer {

  /**
   *
   */
  private val paramsErrorsAccumulator: ParamsErrorsAccumulator = ParamsErrorsAccumulator(this.neuralNetwork)

  /**
   * Normalize the errors, Update the params and reset the errors
   */
  override fun update() {

    if (this.paramsErrorsAccumulator.count >= this.minLossCountToUpdate) {

      this.paramsErrorsAccumulator.averageErrors()
      this.updateParams()
      this.paramsErrorsAccumulator.reset()
    }
  }

  /**
   * Method to call every new epoch.
   *
   * In turn it calls the same method into the `updateMethod`
   */
  override fun newEpoch() {

    if (this.updateMethod is EpochScheduling) {
      this.updateMethod.newEpoch()
    }
  }

  /**
   * Method to call every new batch.
   *
   * In turn it calls the same method into the `updateMethod`
   */
  override fun newBatch() {

    if (this.updateMethod is BatchScheduling) {
      this.updateMethod.newBatch()
    }
  }

  /**
   * Method to call every new example.
   *
   * In turn it calls the same method into the `updateMethod`
   */
  override fun newExample() {

    if (this.updateMethod is ExampleScheduling) {
      this.updateMethod.newExample()
    }
  }

  /**
   *
   */
  fun accumulate(paramsErrors: NetworkParameters) {
    this.paramsErrorsAccumulator.accumulate(paramsErrors)
  }

  /**
   * Update the params respect to the errors using the update helper (Learning Rate, ADAM, AdaGrad, ...)
   */
  private fun updateParams() {

    this.neuralNetwork.model.zip(this.paramsErrorsAccumulator.getParamsErrors()).forEach { (params, errors) ->

      val e = errors.values

      when (e) {
        is DenseNDArray -> this.updateMethod.update(array = params as UpdatableDenseArray, errors = e)
        is SparseNDArray -> this.updateMethod.update(array = params as UpdatableDenseArray, errors = e)
        else -> throw RuntimeException("Invalide errors type")
      }
    }
  }
}
