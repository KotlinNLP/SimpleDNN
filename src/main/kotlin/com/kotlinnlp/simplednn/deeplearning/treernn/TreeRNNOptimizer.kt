/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.treernn

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.learningrate.LearningRateMethod
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParamsOptimizer
import com.kotlinnlp.simplednn.core.optimizer.Optimizer

/**
 * The TreeRNNOptimizer is the optimizer of the TreeRNN which in turn aggregates the optimizers of the sub-networks:
 * leftRNN, rightRNN and concatNetwork.
 *
 * @param network the TreeRNN to optimize
 * @param updateMethod the [UpdateMethod] used to optimize the inner networks
 */
class TreeRNNOptimizer(
  network: TreeRNN,
  updateMethod: UpdateMethod = LearningRateMethod(learningRate = 0.0001)
) : Optimizer(updateMethod) {

  /**
   * The [Optimizer] used for the left recurrent network.
   */
  private val leftOptimizer = NetworkParamsOptimizer(network.leftRNN, updateMethod)

  /**
   * The [Optimizer] used for the right recurrent network.
   */
  private val rightOptimizer = NetworkParamsOptimizer(network.rightRNN, updateMethod)

  /**
   * The [Optimizer] used for the feed-forward concatenation network.
   */
  private val concatOptimizer = NetworkParamsOptimizer(network.concatNetwork, updateMethod)

  /**
   *
   */
  private var accumulatedErrors: Boolean = false

  /**
   * Method to call every new epoch.
   */
  override fun newEpoch() {
    this.leftOptimizer.newEpoch()
    this.rightOptimizer.newEpoch()
    this.concatOptimizer.newEpoch()
  }

  /**
   * Method to call every new batch.
   */
  override fun newBatch() {
    this.leftOptimizer.newBatch()
    this.rightOptimizer.newBatch()
    this.concatOptimizer.newBatch()
  }

  /**
   * Method to call every new example.
   */
  override fun newExample() {
    this.leftOptimizer.newExample()
    this.rightOptimizer.newExample()
    this.concatOptimizer.newExample()
  }

  /**
   * Update the params using the accumulated errors and reset the errors.
   */
  override fun update() {
    if (this.accumulatedErrors) {
      this.leftOptimizer.update()
      this.rightOptimizer.update()
      this.concatOptimizer.update()
      this.accumulatedErrors = false
    }
  }

  /**
   * Accumulate the params errors.
   */
  fun accumulate(errors: TreeRNNParameters) {
    this.leftOptimizer.accumulate(errors.leftRNN)
    this.rightOptimizer.accumulate(errors.rightRNN)
    this.concatOptimizer.accumulate(errors.concatNetwork)
    this.accumulatedErrors = true
  }
}
