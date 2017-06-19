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
abstract class ValidationHelper<ExampleType: Example>(
  open val neuralProcessor: NeuralProcessor,
  val outputEvaluationFunction: OutputEvaluationFunction) {

  /**
   * Validate examples.
   *
   * @param examples a list of examples to validate
   *
   * @return the percentage of correct predictions
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
   * Validate a single example.
   *
   * @param example the example to validate
   *
   * @return a Boolean indicating if the predicted output matches the gold output
   */
  abstract fun validate(example: ExampleType): Boolean
}
