/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.competitivelearning.feedforward

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.optimizer.Optimizer
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The optimizer of the [CLFeedforwardNetworkModel].
 *
 * @param model the CL network model
 * @param updateMethod the update method helper (Learning Rate, ADAM, AdaGrad, ...)
 */
class CLFeedforwardNetworkOptimizer(
  model: CLFeedforwardNetworkModel,
  updateMethod: UpdateMethod<*>
) : Optimizer<Pair<Int, NetworkParameters>>(updateMethod) {

  /**
   * A list of [ParamsOptimizer]s, one for each sub-network.
   */
  private val networksOptimizers: List<ParamsOptimizer<NetworkParameters>> = model.classes.map {
    ParamsOptimizer(params = model.networks[it].model, updateMethod = updateMethod)
  }

  /**
   * Update the parameters of all the sub-networks of the [CLFeedforwardNetworkModel].
   */
  override fun update() {
    this.networksOptimizers.forEach { it.update() }
  }

  /**
   * Accumulate the given [paramsErrors] of the model of a given sub-network.
   *
   * @param paramsErrors a pair with the index of the sub-network to optimize and its params errors
   * @param copy a Boolean indicating if the [paramsErrors] can be used as reference or must be copied.
   *             Set copy = false to optimize the accumulation when the amount of the errors to accumulate is 1.
   *             (default = true)
   */
  override fun accumulate(paramsErrors: Pair<Int, NetworkParameters>, copy: Boolean) {
    this.networksOptimizers[paramsErrors.first].accumulate(paramsErrors.second, copy = copy)
  }
}
