/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.learningrate.LearningRateMethod
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.functionalities.losses.SoftmaxCrossEntropyCalculator
import com.kotlinnlp.simplednn.helpers.training.SequenceTrainingHelper
import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.dataset.*
import com.kotlinnlp.simplednn.core.functionalities.outputevaluation.ClassificationEvaluation
import com.kotlinnlp.simplednn.core.neuralnetwork.preset.CFN
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.helpers.validation.SequenceValidationHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import utils.CorpusReader
import utils.exampleextractor.ClassificationSequenceExampleExtractor

fun main(args: Array<String>) {

  println("Start 'Progressive Sum Test'")

  val dataset = CorpusReader<SequenceExample<DenseNDArray>>().read(
    corpusPath = Configuration.loadFromFile().progressive_sum.datasets_paths, // same for validation and test
    exampleExtractor = ClassificationSequenceExampleExtractor(outputSize = 11),
    perLine = true)

  ProgressiveSumTest(dataset).start()

  println("End.")
}

/**
 *
 */
class ProgressiveSumTest(val dataset: Corpus<SequenceExample<DenseNDArray>>) {

  /**
   *
   */
  private val neuralNetwork = CFN(
      inputSize = 1,
      hiddenSize = 100,
      hiddenActivation = Tanh(),
      outputSize = 11,
      outputActivation = Softmax())

  /**
   *
   */
  fun start() {

    this.initialValidation()
    this.train()
  }

  /**
   *
   */
  private fun initialValidation() {

    println("\n-- VALIDATION BEFORE TRAINING\n")

    val validationHelper = SequenceValidationHelper<DenseNDArray>(
      neuralProcessor = RecurrentNeuralProcessor(this.neuralNetwork, useDropout = false, propagateToInput = false),
      outputEvaluationFunction = ClassificationEvaluation())

    val accuracy: Double = validationHelper.validate(this.dataset.validation)

    println("Accuracy: %.2f%%".format(100.0 * accuracy))
  }

  /**
   *
   */
  private fun train() {

    println("\n-- TRAINING\n")

    val optimizer = ParamsOptimizer(
      params = this.neuralNetwork.model,
      updateMethod = LearningRateMethod(learningRate = 0.1))

    val trainingHelper = SequenceTrainingHelper<DenseNDArray>(
      neuralProcessor = RecurrentNeuralProcessor(this.neuralNetwork, useDropout = false, propagateToInput = false),
      optimizer = optimizer,
      lossCalculator = SoftmaxCrossEntropyCalculator(),
      verbose = true)

    val validationHelper = SequenceValidationHelper<DenseNDArray>(
      neuralProcessor = RecurrentNeuralProcessor(this.neuralNetwork, useDropout = false, propagateToInput = false),
      outputEvaluationFunction = ClassificationEvaluation())

    trainingHelper.train(
      trainingExamples = this.dataset.training,
      validationExamples = this.dataset.validation,
      epochs = 4,
      shuffler = Shuffler(enablePseudoRandom = true, seed = 1),
      batchSize = 1,
      validationHelper = validationHelper)
  }
}
