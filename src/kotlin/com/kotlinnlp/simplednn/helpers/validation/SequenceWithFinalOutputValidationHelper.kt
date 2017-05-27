/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.helpers.validation

import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.dataset.SequenceExampleWithFinalOutput
import com.kotlinnlp.simplednn.core.functionalities.outputevaluation.OutputEvaluationFunction
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 *
 */
class SequenceWithFinalOutputValidationHelper<NDArrayType: NDArray<NDArrayType>>(
  override val neuralProcessor: RecurrentNeuralProcessor<NDArrayType>,
  outputEvaluationFunction: OutputEvaluationFunction
) : ValidationHelper<SequenceExampleWithFinalOutput<NDArrayType>>(
  neuralProcessor = neuralProcessor,
  outputEvaluationFunction = outputEvaluationFunction) {

  override fun validate(example: SequenceExampleWithFinalOutput<NDArrayType>): Boolean {

    val output = this.neuralProcessor.forward(example.sequenceFeatures)
    return this.outputEvaluationFunction(output = output, outputGold = example.outputGold)
  }
}
