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
 * The optimizer of the BiRNN which in turn aggregates the optimizers of its sub-networks: leftToRightNetwork and
 * rightToLeftNetwork.
 *
 * It is recommended to use the same UpdateMethod configuration for each sub-network.
 *
 * @param network the [BiRNN] to optimize
 * @param leftToRightUpdateMethod the [UpdateMethod] used for the left-to-right recurrent network
 * @param rightToLeftUpdateMethod the [UpdateMethod] used for the right-to-left recurrent network
 */
class BiRNNOptimizer(
  network: BiRNN,
  leftToRightUpdateMethod: UpdateMethod = ADAMMethod(stepSize = 0.0001),
  rightToLeftUpdateMethod: UpdateMethod = ADAMMethod(stepSize = 0.0001)
) : Optimizer {

  /**
   * The [ParamsOptimizer] for the left-to-right network.
   */
  private val leftToRightOptimizer = ParamsOptimizer(network.leftToRightNetwork, leftToRightUpdateMethod)

  /**
   * The [ParamsOptimizer] for the right-to-left network.
   */
  private val rightToLeftOptimizer = ParamsOptimizer(network.rightToLeftNetwork, rightToLeftUpdateMethod)

  /**
   * Method to call every new epoch.
   */
  override fun newEpoch() {
    this.leftToRightOptimizer.newEpoch()
    this.rightToLeftOptimizer.newEpoch()
  }

  /**
   * Method to call every new batch.
   */
  override fun newBatch() {
    this.leftToRightOptimizer.newBatch()
    this.rightToLeftOptimizer.newBatch()
  }

  /**
   * Method to call every new example.
   */
  override fun newExample() {
    this.leftToRightOptimizer.newExample()
    this.rightToLeftOptimizer.newExample()
  }

  /**
   * Update the parameters using the accumulated errors and then reset the errors.
   */
  override fun update(): Unit {
    this.leftToRightOptimizer.update()
    this.rightToLeftOptimizer.update()
  }

  /**
   * Accumulate the parameters errors into the optimizer.
   *
   * @param errors the parameters errors to accumulate
   */
  fun accumulate(errors: BiRNNParameters) {
    this.leftToRightOptimizer.accumulate(errors.leftToRight)
    this.rightToLeftOptimizer.accumulate(errors.rightToLeft)
  }
}
