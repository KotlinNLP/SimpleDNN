/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.outputevaluation

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * Evaluation function which returns true if all the binary outputs are equal to the gold binary outputs.
 */
object MulticlassEvaluation : OutputEvaluationFunction {

  /**
   * The evaluation function.
   *
   * @param output the output of a NeuralNetwork
   * @param outputGold the expected gold output
   *
   * @return a Boolean indicating whether the output must be considered equal to the gold or not
   */
  override fun invoke(output: DenseNDArray, outputGold: DenseNDArray): Boolean {
    require(output.length == outputGold.length) { "output and outputGold must have the same dimension" }

    for (i in 0 until outputGold.length) {
      val outputInt = output[i].toInt()
      val outputGoldInt = outputGold[i].toInt()

      require(outputInt == 0 || outputInt == 1) { "non-binary output value" }
      require(outputGoldInt == 0 || outputGoldInt == 1) { "non-binary gold output value" }

      if (outputInt != outputGoldInt) {
        return false
      }
    }

    return true
  }
}
