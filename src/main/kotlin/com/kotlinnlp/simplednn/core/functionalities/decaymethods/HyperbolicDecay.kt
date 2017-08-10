/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.decaymethods

/**
 * HyperbolicDecay defines an hyperbolic decay depending on the time step => LR = LRinit / (1 + decay * t).
 *
 * @property decay the learning rate decay applied at each time step (must be >= 0)
 * @property initLearningRate the initial learning rate (must be >= [finalLearningRate])
 * @property finalLearningRate the final value which the learning rate will reach (must be >= 0)
 */
class HyperbolicDecay(
  val decay: Double,
  val initLearningRate: Double = 0.0,
  val finalLearningRate: Double = 0.0
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
    return if (learningRate > this.finalLearningRate && this.decay > 0.0 && timeStep > 1) {
      this.initLearningRate / (1.0 + this.decay * timeStep)
    } else {
      learningRate
    }
  }
}
