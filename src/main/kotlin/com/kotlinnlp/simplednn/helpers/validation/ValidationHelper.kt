/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.helpers.validation

import com.kotlinnlp.simplednn.dataset.*

/**
 * A helper which executes the validation of a dataset.
 */
abstract class ValidationHelper<ExampleType: Example> {

  /**
   * Validate examples.
   *
   * @param examples a list of examples to validate
   * @param onPrediction an optional callback called for each prediction (args: example, isCorrect)
   * @param saveContributions whether to save the contributions of each input to its output (needed to calculate
   *                          the relevance)
   *
   * @return the percentage of correct predictions
   */
  fun validate(examples: List<ExampleType>,
               onPrediction: (example: ExampleType, isCorrect: Boolean) -> Unit = {_, _ -> },
               saveContributions: Boolean = false): Double {

    var correctPredictions = 0

    examples.forEach {
      val isCorrect = this.validate(it, saveContributions = saveContributions)

      if (isCorrect) {
        correctPredictions += 1
      }

      onPrediction(it, isCorrect)
    }

    return correctPredictions.toDouble() / examples.size
  }

  /**
   * Validate a single example.
   *
   * @param example the example to validate
   * @param saveContributions whether to save the contributions of each input to its output (needed to calculate
   *                          the relevance)
   *
   * @return a Boolean indicating if the predicted output matches the gold output
   */
  abstract fun validate(example: ExampleType, saveContributions: Boolean): Boolean
}
