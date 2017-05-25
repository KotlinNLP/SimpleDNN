/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.outputevaluation

import com.kotlinnlp.simplednn.simplemath.ndarray.DenseNDArray

/**
 * Evaluation function which compares output and output gold as one hot encoders
 */
class ClassificationEvaluation : OutputEvaluationFunction {

  /**
   * The evaluation function.
   *
   * @param output the output of a NeuralNetwork
   * @param outputGold the expected gold output
   *
   * @return a Boolean indicating whether the output must be considered equal to the gold or not
   */
  override fun invoke(output: DenseNDArray, outputGold: DenseNDArray): Boolean {
    require(outputGold.isOneHotEncoder) { "outputGold should be a one hot encoder"}
    return outputGold[output.argMaxIndex()] == 1.0
  }
}
