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
 * @param scorer the [MultiPredictionScorer] to optimize
 * @property updateMethod the update method helper (Learning Rate, ADAM, AdaGrad, ...)
 */
class MultiPredictionOptimizer(
  scorer: MultiPredictionScorer<*>,
  updateMethod: UpdateMethod<*>
) : Optimizer(updateMethod) {

  /**
   * A list of [ParamsOptimizer]s, one for each sub-network.
   */
  private val networksOptimizers: List<ParamsOptimizer<NetworkParameters>> = List(
    size = scorer.model.networks.size,
    init = { i -> ParamsOptimizer(params = scorer.model.networks[i].model, updateMethod = updateMethod) }
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
   * @param networksErrors a map of sub-networks indices to their params errors
   * @param copy a Boolean indicating if the [networksErrors] can be used as reference or must be copied.
   *             Set copy = false to optimize the accumulation when the amount of the errors to accumulate is 1.
   *             (default = true)
   */
  fun accumulate(networksErrors: Map<Int, NetworkParameters>, copy: Boolean = true) {

    networksErrors.forEach { networkIndex, errors ->
      this.networksOptimizers[networkIndex].accumulate(errors, copy = copy)
    }
  }
}
