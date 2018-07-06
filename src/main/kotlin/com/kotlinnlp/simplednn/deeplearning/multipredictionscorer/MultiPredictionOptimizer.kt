/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.multipredictionscorer

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod

import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.optimizer.Optimizer
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer

/**
 * The [Optimizer] of the MultiPredictionScorer.
 *
 * @param model the [MultiPredictionModel] to optimize
 * @property updateMethod the update method helper (Learning Rate, ADAM, AdaGrad, ...)
 */
class MultiPredictionOptimizer(
  model: MultiPredictionModel,
  updateMethod: UpdateMethod<*>
) : Optimizer<Map<Int, NetworkParameters>>(updateMethod) {

  /**
   * A list of [ParamsOptimizer]s, one for each sub-network.
   */
  private val networksOptimizers: List<ParamsOptimizer<NetworkParameters>> = List(
    size = model.networks.size,
    init = { i -> ParamsOptimizer(params = model.networks[i].model, updateMethod = updateMethod) }
  )

  /**
   * Update the parameters of all the sub-networks of the [MultiPredictionScorer].
   */
  override fun update() {
    this.networksOptimizers.forEach { it.update() }
  }

  /**
   * Accumulate the given [networksErrors] for each sub-network of the [MultiPredictionScorer].
   *
   * @param paramsErrors a map of sub-networks indices to their params errors
   * @param copy a Boolean indicating if the [networksErrors] can be used as reference or must be copied.
   *             Set copy = false to optimize the accumulation when the amount of the errors to accumulate is 1.
   *             (default = true)
   */
  override fun accumulate(paramsErrors: Map<Int, NetworkParameters>, copy: Boolean) {

    paramsErrors.forEach { networkIndex, errors ->
      this.networksOptimizers[networkIndex].accumulate(errors, copy = copy)
    }
  }

  /**
   * Accumulate the given [errors] of the model sub-network with the given [networkIndex].
   *
   * @param networkIndex the index of a network of the model
   * @param errors the errors of the network with the given [networkIndex]
   * @param copy a Boolean indicating if the [errors] can be used as reference or must be copied.
   *             Set copy = false to optimize the accumulation when the amount of the errors to accumulate is 1.
   *             (default = true)
   */
  fun accumulate(networkIndex: Int, errors: NetworkParameters, copy: Boolean = true) {
    this.networksOptimizers[networkIndex].accumulate(errors, copy = copy)
  }
}
