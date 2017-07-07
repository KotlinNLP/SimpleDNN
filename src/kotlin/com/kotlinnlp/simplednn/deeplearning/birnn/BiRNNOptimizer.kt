/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.birnn

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.core.optimizer.Optimizer
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod

/**
 * The BiRNNOptimizer is the optimizer of the BiRNN which in turn aggregate
 * the optimizers of its sub-networks: leftToRightNetwork, rightToLeftNetwork, outputNetwork.
 *
 * It is recommended to use the same UpdateMethod configuration for all the sub-networks.
 *
 * @param network the TreeRNN to optimize
 * @param leftToRightUpdateMethod the [UpdateMethod] used for the left-to-right recurrent network
 * @param rightToLeftUpdateMethod the [UpdateMethod] used for the right-to-left recurrent network
 * @param outputUpdateMethod the [UpdateMethod] used for the feed-forward output network
 */
class BiRNNOptimizer(
  network: BiRNN,
  leftToRightUpdateMethod: UpdateMethod = ADAMMethod(stepSize = 0.0001),
  rightToLeftUpdateMethod: UpdateMethod = ADAMMethod(stepSize = 0.0001),
  outputUpdateMethod: UpdateMethod = ADAMMethod(stepSize = 0.0001)
) : Optimizer {

  /**
   * The [Optimizer] used for the left-to-right network
   */
  private val leftToRightOptimizer = ParamsOptimizer(network.leftToRightNetwork, leftToRightUpdateMethod)

  /**
   * The [Optimizer] used for the right-to-left network
   */
  private val rightToLeftOptimizer = ParamsOptimizer(network.rightToLeftNetwork, rightToLeftUpdateMethod)

  /**
   * The [Optimizer] used for the output network
   */
  private val outputNetwork = ParamsOptimizer(network.outputNetwork, outputUpdateMethod)

  /**
   * Method to call every new epoch.
   */
  override fun newEpoch() {
    this.leftToRightOptimizer.newEpoch()
    this.rightToLeftOptimizer.newEpoch()
    this.outputNetwork.newEpoch()
  }

  /**
   * Method to call every new batch.
   */
  override fun newBatch() {
    this.leftToRightOptimizer.newBatch()
    this.rightToLeftOptimizer.newBatch()
    this.outputNetwork.newBatch()
  }

  /**
   * Method to call every new example.
   */
  override fun newExample() {
    this.leftToRightOptimizer.newExample()
    this.rightToLeftOptimizer.newExample()
    this.outputNetwork.newExample()
  }

  /**
   * Update the params using the accumulated errors and reset the errors
   */
  override fun update(): Unit {
    this.leftToRightOptimizer.update()
    this.rightToLeftOptimizer.update()
    this.outputNetwork.update()
  }

  /**
   * Accumulate the params errors on the optimizer
   *
   * @param errors params errors to accumulate
   */
  fun accumulate(errors: BiRNNParameters) {
    this.leftToRightOptimizer.accumulate(errors.leftToRight)
    this.rightToLeftOptimizer.accumulate(errors.rightToLeft)
    this.outputNetwork.accumulate(errors.output)
  }
}
