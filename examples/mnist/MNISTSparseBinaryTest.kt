/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package mnist

import com.kotlinnlp.simplednn.core.arrays.DistributionArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ELU
import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.decaymethods.HyperbolicDecay
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.learningrate.LearningRateMethod
import com.kotlinnlp.simplednn.core.neuralnetwork.preset.FeedforwardNeuralNetwork
import traininghelpers.training.FeedforwardTrainer
import com.kotlinnlp.simplednn.core.functionalities.outputevaluation.ClassificationEvaluation
import traininghelpers.validation.FeedforwardEvaluator
import com.kotlinnlp.simplednn.core.functionalities.losses.SoftmaxCrossEntropyCalculator
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray
import utils.Corpus
import utils.SimpleExample
import mnist.helpers.MNISTSparseExampleExtractor
import utils.CorpusReader

fun main() {

  println("Start 'MNIST Sparse Binary Test'")

  val dataset = CorpusReader<SimpleExample<SparseBinaryNDArray>>().read(
    corpusPath = Configuration.loadFromFile().mnist.datasets_paths,
    exampleExtractor = MNISTSparseExampleExtractor(outputSize = 10),
    perLine = false)

  MNISTSparseBinaryTest(dataset).start()

  println("\nEnd.")
}

/**
 *
 */
private class MNISTSparseBinaryTest(private val dataset: Corpus<SimpleExample<SparseBinaryNDArray>>) {

  /**
   *
   */
  private val neuralNetwork = FeedforwardNeuralNetwork(
    inputSize = 784,
    inputType = LayerType.Input.SparseBinary,
    hiddenSize = 100,
    hiddenActivation = ELU(),
    outputSize = 10,
    outputActivation = Softmax())

  /**
   *
   */
  fun start() {

    this.train()

    // TODO: Implement forward with contributions also for sparse binary inputs
    // this.printImages(examples = ArrayList(this.dataset.validation.subList(0, 20))) // reduced sublist
  }

  /**
   *
   */
  private fun train() {

    println("\n-- TRAINING\n")

    FeedforwardTrainer(
      model = this.neuralNetwork,
      updateMethod = LearningRateMethod(
        learningRate = 0.01,
        decayMethod = HyperbolicDecay(decay = 0.5, initLearningRate = 0.01)),
      lossCalculator = SoftmaxCrossEntropyCalculator(),
      examples = this.dataset.training,
      epochs = 3,
      batchSize = 1,
      evaluator = FeedforwardEvaluator(
        model = this.neuralNetwork,
        examples = this.dataset.validation,
        outputEvaluationFunction = ClassificationEvaluation())
    ).train()
  }

  /**
   *
   */
  private fun printImages(examples: ArrayList<SimpleExample<SparseBinaryNDArray>>) {

    println("\n-- PRINT IMAGES RELEVANCE\n")

    FeedforwardEvaluator(
      model = this.neuralNetwork,
      examples = examples,
      outputEvaluationFunction = ClassificationEvaluation(),
      saveContributions = true,
      afterEachEvaluation = { example, _, processor ->

        val sparseRelevance = processor.calculateInputRelevance(DistributionArray.uniform(length = 10))
        val denseRelevance: DenseNDArray = DenseNDArrayFactory.zeros(Shape(784)).assignValues(sparseRelevance)

        this.printImage(image = denseRelevance, value = example.outputGold.argMaxIndex())
      }
    ).evaluate()
  }

  /**
   *
   */
  private fun printImage(image: DenseNDArray, value: Int) {

    println("------------------ %d -----------------".format(value))

    for (i in 0 until 28) {
      for (j in 0 until 28) {
        print(if (image[i * 28 + j] > 0.0) "# " else "  ")
      }
      println()
    }
  }
}
