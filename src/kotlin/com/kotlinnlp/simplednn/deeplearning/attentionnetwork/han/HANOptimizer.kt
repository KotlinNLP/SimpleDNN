/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attentionnetwork.han

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.core.optimizer.Optimizer
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.AttentionNetworkOptimizer
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNOptimizer
import com.kotlinnlp.simplednn.utils.scheduling.BatchScheduling
import com.kotlinnlp.simplednn.utils.scheduling.EpochScheduling
import com.kotlinnlp.simplednn.utils.scheduling.ExampleScheduling

/**
 * The optimizer of the HAN which in turn aggregates the optimizers of the networks of each hierarchical level: BiRNNs,
 * AttentionNetworks and the Feedforward output model.
 *
 * @param model the [HAN] model to optimize
 * @param updateMethod the [UpdateMethod] used for all the inner networks
 */
class HANOptimizer(model: HAN, val updateMethod: UpdateMethod = ADAMMethod(stepSize = 0.0001)) : Optimizer {

  /**
   * The [ParamsOptimizer] for the BiRNNs.
   */
  private val biRNNsOptimizers = Array(
    size = model.hierarchySize,
    init = { i -> BiRNNOptimizer(network = model.biRNNs[i], updateMethod = this.updateMethod) }
  )

  /**
   * The [ParamsOptimizer] for the AttentionNetworks.
   */
  private val attentionNetworksOptimizers = Array(
    size = model.hierarchySize,
    init = { i ->
      AttentionNetworkOptimizer(model = model.attentionNetworksParams[i], updateMethod = this.updateMethod)
    }
  )

  /**
   * The [ParamsOptimizer] for the output Feedforward network.
   */
  private val outputOptimizer = ParamsOptimizer(model.outputNetwork, updateMethod)

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
   * Update the parameters using the accumulated errors and then reset the errors.
   */
  override fun update(): Unit {
    this.biRNNsOptimizers.forEach { it.update() }
    this.attentionNetworksOptimizers.forEach { it.update() }
    this.outputOptimizer.update()
  }

  /**
   * Accumulate the parameters errors into the optimizers.
   *
   * @param errors the parameters errors to accumulate
   */
  fun accumulate(errors: HANParameters) {

    this.biRNNsOptimizers.forEachIndexed { i, optimizer -> optimizer.accumulate(errors.biRNNs[i]) }
    this.attentionNetworksOptimizers.forEachIndexed { i, optimizer ->
      optimizer.accumulate(errors.attentionNetworks[i])
    }
    this.outputOptimizer.accumulate(errors.outputNetwork)
  }
}
