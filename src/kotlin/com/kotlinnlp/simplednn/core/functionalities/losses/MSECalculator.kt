/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.losses

import com.kotlinnlp.simplednn.simplemath.*

/**
 *
 * MeanSquaredErrorCalculator
 */
open class MSECalculator : LossCalculator {

  /**
   * Loss Derivative
   *
   * @param output  current output layer
   * @param outputGold expected binary output
   * @return calculated avgLoss
   */
  override fun calculateLoss(output: NDArray, outputGold: NDArray): NDArray {
    return output.sub(outputGold).pow(2.0).prod(0.5)
  }

  /**
   * @param output  current output layer
   * @param outputGold expected binary output
   * @return calculated avgLoss
   */
  override fun calculateErrors(output: NDArray, outputGold: NDArray): NDArray {
    return output.sub(outputGold)
  }

}
