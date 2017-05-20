/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.decaymethods

/**
 *
 * @param decay                Double >= 0. Learning rate decay over each update.
 * @param initLearningRate     Double >= 0. Initial Learning rate
 * @param finalLearningRate    Double >= 0. Learning rate beyond which it is no longer applied the decay
 */
class HyperbolicDecay(
  val decay: Double,
  val initLearningRate: Double = 0.0,
  val finalLearningRate: Double = 0.0
) : DecayMethod {

  /**
   *
   */
  init { require(initLearningRate > finalLearningRate) }

  /**
   *
   * @param learningRate current learning rate
   * @param timeStep time step
   * @return decayed learning rate
   */
  fun update(learningRate: Double, timeStep: Int): Double {
    return if (learningRate > this.finalLearningRate && this.decay > 0.0 && timeStep > 1) {
      this.initLearningRate / (1.0 + this.decay * timeStep)
    } else {
      learningRate
    }
  }
}
