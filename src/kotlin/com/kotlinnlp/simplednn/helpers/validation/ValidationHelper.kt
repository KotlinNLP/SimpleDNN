/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.helpers.validation

import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.dataset.*
import com.kotlinnlp.simplednn.core.functionalities.outputevaluation.OutputEvaluationFunction

/**
 *
 * @param neuralProcessor neuralProcessor
 */
abstract class ValidationHelper<ExampleType: Example>(open val neuralProcessor: NeuralProcessor,
                                                      val outputEvaluationFunction: OutputEvaluationFunction) {

  /**
   *
   * @param examples example
   * @return percentage of correct predictions
   */
  fun validate(examples: ArrayList<ExampleType>): Double {

    var correctPredictions: Int = 0

    examples.forEach {
      if (this.validate(it)) {
        correctPredictions += 1
      }
    }

    return correctPredictions.toDouble() / examples.size
  }
  /**
   *
   * @param example a single example
   * @return true if the output match the gold output, false otherwise
   */
  abstract fun validate(example: ExampleType): Boolean
}
