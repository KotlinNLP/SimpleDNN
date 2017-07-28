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
import com.kotlinnlp.simplednn.utils.scheduling.BatchScheduling
import com.kotlinnlp.simplednn.utils.scheduling.EpochScheduling
import com.kotlinnlp.simplednn.utils.scheduling.ExampleScheduling

/**
 * The optimizer of the BiRNN which in turn aggregates the optimizers of its sub-networks: leftToRightNetwork and
 * rightToLeftNetwork.
 *
 * @param network the [BiRNN] to optimize
 * @param updateMethod the [UpdateMethod] used for the left-to-right and right-to-left recurrent networks
 */
class BiRNNOptimizer(
  network: BiRNN,
  val updateMethod: UpdateMethod = ADAMMethod(stepSize = 0.0001)
) : Optimizer {

  /**
   * The [ParamsOptimizer] for the left-to-right network.
   */
  private val leftToRightOptimizer = ParamsOptimizer(network.leftToRightNetwork, updateMethod)

  /**
   * The [ParamsOptimizer] for the right-to-left network.
   */
  private val rightToLeftOptimizer = ParamsOptimizer(network.rightToLeftNetwork, updateMethod)

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
