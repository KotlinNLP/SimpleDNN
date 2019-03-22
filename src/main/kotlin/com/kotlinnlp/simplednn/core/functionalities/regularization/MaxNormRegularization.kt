/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.regularization

import com.kotlinnlp.simplednn.core.arrays.ParamsArray

/**
 * Regularize weights before the update
 *
 * @property lambda regularization parameter
 */
class MaxNormRegularization(override val lambda: Double) : WeightsRegularization {

  /**
   * @param weights the weights to regularize
   */
  override fun apply(weights: ParamsArray) {

    val norm2 = weights.values.norm2()

    (0 until weights.values.length)
      .filter { weights.values[it] > norm2 }
      .forEach { weights.values[it] = norm2 }
  }
}
