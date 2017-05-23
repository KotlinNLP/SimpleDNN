/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.decaymethods

/**
 *
 * @param initLearningRate Double >= 0. Initial Learning rate
 * @param finalLearningRate Double >= 0. Learning rate beyond which it is no longer applied the decay
 * @param totalIterations Double >= 0. Total of iterations
 */
class ExponentialDecay(
  val totalIterations: Int,
  val initLearningRate: Double = 0.0,
  val finalLearningRate: Double = 0.0
) : DecayMethod {

  /**
   *
   */
  init { require(initLearningRate > finalLearningRate) }

  /**
   *
   * @param learningRate learningRate
   * @param timeStep time step
   * @return decayed learning rate
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
