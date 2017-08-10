/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.decaymethods

/**
 * An interface which defines the decay method used to decrease the learning rate of an UpdateMethod, given
 * a scheduled time step.
 */
interface DecayMethod {

  /**
   * Update the learning rate given a time step.
   *
   * @param learningRate the learning rate to decrease
   * @param timeStep the current time step
   *
   * @return the updated learning rate
   */
  fun update(learningRate: Double, timeStep: Int): Double
}
