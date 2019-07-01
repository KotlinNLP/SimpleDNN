/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.losses

import kotlin.math.exp
import kotlin.math.ln

/**
 * TODO: move the class into a more appropriate package (part of the 'loss' refactoring)
 *
 * @param predictions the predictions
 * @param goldIndex the index of the gold item in the predictions
 */
class NegativeLogProbability(
  private val predictions: List<Double>,
  private val goldIndex: Int
) {

  /**
   * The sum of the exponentials.
   */
  private val sumExp: Double by lazy { this.predictions.map { exp(it) }.sum() }

  /**
   * @return the loss
   */
  fun f(): Double = -this.predictions[this.goldIndex] + ln(this.sumExp)

  /**
   * @return the gradients
   */
  fun df(): List<Double> {

    val c = 1.0 / this.sumExp

    return this.predictions.mapIndexed { i, xi ->
      if (i == this.goldIndex)
        c * exp(xi) - 1.0
      else
        c * exp(xi)
    }
  }
}
