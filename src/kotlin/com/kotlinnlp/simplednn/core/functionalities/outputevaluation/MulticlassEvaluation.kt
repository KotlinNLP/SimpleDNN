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
class MulticlassEvaluation : OutputEvaluationFunction {
  /**
   *
   * @param output current output
   * @param outputGold outputGold
   * @return Boolean
   */
  override fun invoke(output: NDArray, outputGold: NDArray): Boolean {
    require(outputGold.length == outputGold.length) { "outputLayer and outputGold must have the same dimension" }

    val binaryOutput: NDArray = output.roundInt()

    for (i in 0 until outputGold.length) {
      require(binaryOutput[i].toInt() == 1 || binaryOutput[i].toInt() == 0) { "non-binary output value" }
      require(outputGold[i].toInt() == 1 || outputGold[i].toInt() == 0) { "non-binary gold output value" }

      if (binaryOutput[i].toInt() != outputGold[i].toInt()) return false
    }

    return true
  }

}
