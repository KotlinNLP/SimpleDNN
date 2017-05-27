/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.regularization

import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray

/**
 * Regularize weights before the update
 *
 * @property lambda regularization parameter
 */
class L1Regularization(override val lambda: Double) : WeightsRegularization {

  /**
   * w = w - sign(w) * lambda
   *
   * @param weights the weights to regularize
   */
  override fun apply(weights: UpdatableDenseArray) {
    weights.values.assignSub(weights.values.sign().assignProd(lambda))
  }
}
