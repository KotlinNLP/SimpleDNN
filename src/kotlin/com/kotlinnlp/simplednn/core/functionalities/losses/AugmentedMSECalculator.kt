/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.losses

import com.kotlinnlp.simplednn.simplemath.NDArray

/**
 * Augmented Mean Squared Error calculator
 *
 * @property pi pi (0.1 by default)
 * @property c c (10 by default)
 */
class AugmentedMSECalculator(val pi: Double = 0.1, val c: Double = 10.0) : LossCalculator {

  /**
   *
   */
  var injectedError: Double = 0.0

  /**
   *
   */
  private val isLossPartitionEnabled: Boolean = pi != 0.0

  /**
   *
   */
  private val lossPartition: Double = 1.0 - pi

  /**
   * Calculate the loss within an output and its gold.
   *
   * @param output  current output layer
   * @param outputGold expected binary output
   *
   * @return the loss within [output] and [outputGold]
   */
  override fun calculateLoss(output: NDArray, outputGold: NDArray): NDArray {

    if (!this.isLossPartitionEnabled) {
      return output.sub(outputGold).assignPow(2.0).assignProd(0.5)

    } else {
      val lossContribute = outputGold.sum(output).assignPow(2.0)
      val injectedContribute = output.prod(this.calculateRegularization()).assignPow(2.0)

      // 0.5 * ((1 - pi) * (g - o)^2 + pi * (o * reg)^2)
      return lossContribute.assignProd(this.lossPartition)
        .assignSum(injectedContribute.assignProd(this.pi))
        .assignProd(0.5)
    }
  }

  /**
   * Calculate the errors within an output and its gold.
   *
   * @param output current output layer
   * @param outputGold expected binary output
   *
   * @return the derivative of the loss within [output] and [outputGold]
   */
  override fun calculateErrors(output: NDArray, outputGold: NDArray): NDArray {

    if (!this.isLossPartitionEnabled) {
      return output.sub(outputGold)

    } else {
      val regularization: Double = this.calculateRegularization()

      // (1 - pi) * (o - g) + pi * (o * reg)
      return output.sub(outputGold).assignProd(this.lossPartition)
        .assignSum(
          output.prod(regularization).assignProd(pi))
    }
  }

  /**
   *
   */
  private fun calculateRegularization(): Double = 1.0 - Math.exp(- c * this.injectedError)
}

