/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.activations.Softsign
import com.kotlinnlp.simplednn.core.functionalities.losses.SoftmaxCrossEntropyCalculator
import com.kotlinnlp.simplednn.core.functionalities.outputevaluation.ClassificationEvaluation
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adagrad.AdaGradMethod
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.neuralnetwork.preset.FeedforwardNeuralNetwork
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.core.optimizer.GenericParamsOptimizer
import utils.Corpus
import utils.SimpleExample
import traininghelpers.training.FeedforwardTrainingHelper
import traininghelpers.validation.FeedforwardValidationHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray
import com.kotlinnlp.utils.Shuffler
import utils.CorpusReader
import utils.exampleextractor.ClassificationSparseExampleExtractor

fun main(args: Array<String>) {

  println("Start 'Sparse Input Test'")

  val dataset = CorpusReader<SimpleExample<SparseBinaryNDArray>>().read(
    corpusPath = Configuration.loadFromFile().sparse_input.datasets_paths,
    exampleExtractor = ClassificationSparseExampleExtractor(inputSize = 356425, outputSize = 86),
    perLine = true)

  SparseInputTest(dataset).start()

  println("End.")
}

/**
 *
 */
class SparseInputTest(val dataset: Corpus<SimpleExample<SparseBinaryNDArray>>) {

  /**
   *
   */
  private val neuralNetwork = FeedforwardNeuralNetwork(
    inputSize = 356425,
    inputType = LayerType.Input.SparseBinary,
    hiddenSize = 200,
    hiddenActivation = Softsign(),
    outputSize = 86,
    outputActivation = Softmax())

  /**
   *
   */
  fun start() {
    this.train()
  }

  /**
   *
   */
  private fun train() {

    println("\n-- TRAINING")

    val optimizer = GenericParamsOptimizer(updateMethod = AdaGradMethod(learningRate = 0.1))

    val trainingHelper = FeedforwardTrainingHelper<SparseBinaryNDArray>(
      neuralProcessor = FeedforwardNeuralProcessor(this.neuralNetwork, useDropout = false, propagateToInput = false),
      optimizer = optimizer,
      lossCalculator = SoftmaxCrossEntropyCalculator(),
      verbose = true)

    val validationHelper = FeedforwardValidationHelper<SparseBinaryNDArray>(
      neuralProcessor = FeedforwardNeuralProcessor(this.neuralNetwork, useDropout = false, propagateToInput = false),
      outputEvaluationFunction = ClassificationEvaluation())

    trainingHelper.train(
      trainingExamples = this.dataset.training,
      validationExamples = this.dataset.validation,
      epochs = 3,
      batchSize = 1,
      shuffler = Shuffler(enablePseudoRandom = true, seed = 1),
      validationHelper = validationHelper)
  }
}
