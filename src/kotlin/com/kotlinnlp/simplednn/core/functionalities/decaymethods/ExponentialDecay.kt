/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.decaymethods

/**
 * ExponentialDecay defines an exponential decay depending on the time step
 * => LR = exp((iterations - t) * log(LR) + log(LRfinal))
 *
 * @property initLearningRate the initial Learning rate (must be >= [finalLearningRate])
 * @property finalLearningRate the final value which the learning rate will reach (must be >= 0)
 * @property totalIterations total amount of iterations (must be >= 0)
 */
class ExponentialDecay(
  val initLearningRate: Double = 0.0,
  val finalLearningRate: Double = 0.0,
  val totalIterations: Int
) : DecayMethod {

  /**
   *
   */
  init { require(this.initLearningRate > this.finalLearningRate) }

  /**
   * Update the learning rate given a time step.
   *
   * @param learningRate the learning rate to decrease
   * @param timeStep the current time step
   *
   * @return the updated learning rate
   */
  override fun update(learningRate: Double, timeStep: Int): Double {
    return if (learningRate > this.finalLearningRate && timeStep > 1) {
      Math.exp(
        ((this.totalIterations - timeStep) * Math.log(learningRate) + Math.log(this.finalLearningRate))
          /
          (this.totalIterations - timeStep + 1))
    } else {
      learningRate
    }
  }
}
