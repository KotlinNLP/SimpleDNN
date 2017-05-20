/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.outputevaluation

import com.kotlinnlp.simplednn.simplemath.NDArray


/**
 *
 */
class ClassificationEvaluation : OutputEvaluationFunction {
  /**
   *
   * @param output current output
   * @param outputGold outputGold
   * @return Boolean
   */
  override fun invoke(output: NDArray, outputGold: NDArray): Boolean {
    require(outputGold.isOneHotEncoder) { "outputGold should be a one hot encoder"}
    return outputGold[output.argMaxIndex()] == 1.0
  }
}
