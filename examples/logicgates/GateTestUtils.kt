/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package logicgates

import com.kotlinnlp.simplednn.core.functionalities.activations.ELU
import com.kotlinnlp.simplednn.core.functionalities.activations.Sigmoid
import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import traininghelpers.training.FeedforwardTrainer
import com.kotlinnlp.simplednn.core.functionalities.decaymethods.HyperbolicDecay
import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.learningrate.LearningRateMethod
import com.kotlinnlp.simplednn.core.functionalities.losses.SoftmaxCrossEntropyCalculator
import com.kotlinnlp.simplednn.core.neuralnetwork.preset.FeedforwardNeuralNetwork
import utils.SimpleExample
import com.kotlinnlp.simplednn.core.functionalities.outputevaluation.ClassificationEvaluation
import com.kotlinnlp.simplednn.core.functionalities.outputevaluation.MulticlassEvaluation
import com.kotlinnlp.simplednn.core.functionalities.outputevaluation.OutputEvaluationFunction
import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import traininghelpers.validation.FeedforwardEvaluator
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

object GateTestUtils {

  /**
   *
   */
  fun testAccuracyWithSoftmax(inputSize: Int, examples: ArrayList<SimpleExample<DenseNDArray>>, epochs: Int): Double {

    val neuralNetwork = FeedforwardNeuralNetwork(
      inputSize = inputSize,
      hiddenSize = 10,
      hiddenActivation = ELU(),
      outputSize = 2,
      outputActivation = Softmax())

    return this.testAccuracy(
      neuralNetwork = neuralNetwork,
      examples = examples,
      evaluationFunction = ClassificationEvaluation(),
      epochs = epochs)
  }

  /**
   *
   */
  fun testAccuracyWithSigmoid(inputSize: Int, examples: ArrayList<SimpleExample<DenseNDArray>>, epochs: Int): Double {

    val neuralNetwork = FeedforwardNeuralNetwork(
      inputSize = inputSize,
      hiddenSize = 10,
      hiddenActivation = ELU(),
      outputSize = 1,
      outputActivation = Sigmoid)

    return this.testAccuracy(
      neuralNetwork = neuralNetwork,
      examples = examples,
      evaluationFunction = MulticlassEvaluation(),
      epochs = epochs)
  }

  /**
   *
   */
  private fun testAccuracy(neuralNetwork: StackedLayersParameters,
                           examples: ArrayList<SimpleExample<DenseNDArray>>,
                           evaluationFunction: OutputEvaluationFunction,
                           epochs: Int): Double {

    val trainer = FeedforwardTrainer(
      model = neuralNetwork,
      updateMethod = LearningRateMethod(
        learningRate = 0.01,
        decayMethod = HyperbolicDecay(decay = 0.0, initLearningRate = 0.01)),
      lossCalculator = if (evaluationFunction is ClassificationEvaluation)
        SoftmaxCrossEntropyCalculator
      else
        MSECalculator(),
      examples = examples,
      epochs = epochs,
      batchSize = 1,
      evaluator = FeedforwardEvaluator(
        model = neuralNetwork,
        examples = examples,
        outputEvaluationFunction = evaluationFunction,
        verbose = false),
      verbose = false)

    trainer.train()

    return trainer.bestAccuracy
  }
}
