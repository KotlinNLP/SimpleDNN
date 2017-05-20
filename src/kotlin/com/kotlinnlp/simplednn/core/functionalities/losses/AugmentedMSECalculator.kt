/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.losses

import com.kotlinnlp.simplednn.simplemath.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape

/**
 *
 * AugmentedMeanSquaredErrorCalculator
 * @param pi pi (0.1 default value)
 * @param c c (10 default value)
 */
class AugmentedMSECalculator(val pi: Double = 0.1, val c: Double = 10.0) : LossCalculator {

  /**
   *
   */
  var augmentedError: Double = 0.0

  /**
   *
   */
  private val isLossPartitionEnabled: Boolean = pi != 0.0

  /**
   *
   */
  private val lossPartition: Double = 1.0 - pi

  /**
   *
   */
  private fun calculateRegularization(): Double = 1.0 - Math.exp(- c * this.augmentedError)

  /**
   * @param output  current output layer
   * @param outputGold expected binary output
   * @return calculated avgLoss
   */
  override fun calculateLoss(output: NDArray, outputGold: NDArray): NDArray {

    val loss: NDArray = NDArray.emptyArray(Shape(output.length))

    if (!this.isLossPartitionEnabled) {

      loss.assignValues(output.sub(outputGold).pow(2.0).prod(0.5))

    } else {

      val regularization: Double = this.calculateRegularization()

      for (i in 0 until output.length) {
        val o: Double = output[i].toDouble()
        val g: Double = outputGold[i].toDouble()

        val lossContribute: Double = Math.pow(g - o, 2.0)
        val augmentedContribute: Double = Math.pow(regularization * o, 2.0)

        loss[i] = 0.5 * (this.lossPartition * lossContribute + pi * augmentedContribute)
      }

    }

    return loss
  }

  /**
   *
   * Loss Derivative
   *
   * @param output  current output layer
   * @param outputGold expected binary output
   * @return calculated avgLoss
   */
  override fun calculateErrors(output: NDArray, outputGold: NDArray): NDArray {

    val lossDerivative: NDArray = NDArray.emptyArray(Shape(output.length))

    if (!this.isLossPartitionEnabled) {

      lossDerivative.assignValues(output.sub(outputGold))

    } else {

      val regularization: Double = this.calculateRegularization()

      for (i in 0 until output.length) {
        val o: Double = output[i].toDouble()
        val g: Double = outputGold[i].toDouble()

        lossDerivative[i] = (this.lossPartition * (o - g)) + (pi * o * regularization)
      }
    }

    return lossDerivative
  }

}

