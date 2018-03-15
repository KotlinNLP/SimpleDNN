/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.competitivelearning

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.optimizer.Optimizer
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer

/**
 * The optimizer of the [CLNetworkModel].
 *
 * @property model the network model
 */
class CLNetworkOptimizer(
  private val model: CLNetworkModel,
  updateMethod: UpdateMethod<*>
) : Optimizer(updateMethod) {

  /**
   * A list of [ParamsOptimizer]s, one for each sub-network.
   */
  private val networksOptimizers: Map<Int, ParamsOptimizer<NetworkParameters>> = this.model.classes.associate {
    it to ParamsOptimizer(params = model.networks.getValue(it).model, updateMethod = updateMethod)
  }

  /**
   * Update the parameters of all the sub-networks of the [CLNetworkModel].
   */
  override fun update() {
    this.networksOptimizers.values.forEach { it.update() }
  }

  /**
   * Accumulate the given [errors] of the model sub-network with the given [classId].
   *
   * @param classId the class
   * @param errors the errors
   * @param copy a Boolean indicating if the [errors] can be used as reference or must be copied.
   *             Set copy = false to optimize the accumulation when the amount of the errors to accumulate is 1.
   *             (default = true)
   */
  fun accumulate(classId: Int, errors: NetworkParameters, copy: Boolean = true) {
    this.networksOptimizers[classId]!!.accumulate(errors, copy = copy)
  }

  /**
   * Accumulate the given [errors] of the model sub-network with the given classId.
   *
   * @param errors the pair of classId and errors
   * @param copy a Boolean indicating if the [errors] values can be used as reference or must be copied.
   *             Set copy = false to optimize the accumulation when the amount of the errors to accumulate is 1.
   *             (default = true)
   */
  fun accumulate(errors: Pair<Int, NetworkParameters>, copy: Boolean = true) {
    this.networksOptimizers[errors.first]!!.accumulate(errors.second, copy = copy)
  }
}
