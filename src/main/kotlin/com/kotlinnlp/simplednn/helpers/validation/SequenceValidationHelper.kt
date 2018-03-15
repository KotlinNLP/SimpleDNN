/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.helpers.validation

import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.dataset.SequenceExample
import com.kotlinnlp.simplednn.core.functionalities.outputevaluation.OutputEvaluationFunction
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * A helper which executes the validation of a dataset of [SequenceExample]s.
 *
 * @property neuralProcessor a recurrent neural processor
 * @property outputEvaluationFunction the output evaluation function
 */
class SequenceValidationHelper<NDArrayType: NDArray<NDArrayType>>(
  val neuralProcessor: RecurrentNeuralProcessor<NDArrayType>,
  val outputEvaluationFunction: OutputEvaluationFunction
) : ValidationHelper<SequenceExample<NDArrayType>>() {

  /**
   * Validate a single example.
   *
   * @param example the example to validate
   * @param saveContributions whether to save the contributions of each input to its output (needed to calculate
   *                          the relevance)
   *
   * @return a Boolean indicating if the predicted output matches the gold output
   */
  override fun validate(example: SequenceExample<NDArrayType>, saveContributions: Boolean): Boolean {

    val output = this.neuralProcessor.forward(example.sequenceFeatures, saveContributions = saveContributions)

    return this.outputEvaluationFunction(output = output, outputGold = example.sequenceOutputGold.last())
  }
}
