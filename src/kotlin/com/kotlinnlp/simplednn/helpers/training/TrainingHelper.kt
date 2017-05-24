/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.helpers.training

import com.kotlinnlp.simplednn.core.functionalities.losses.LossCalculator
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.helpers.training.utils.TrainingStatistics
import com.kotlinnlp.simplednn.helpers.validation.ValidationHelper
import com.kotlinnlp.simplednn.dataset.Example
import com.kotlinnlp.simplednn.dataset.Shuffler
import com.kotlinnlp.simplednn.helpers.training.utils.ExamplesIndices
import com.kotlinnlp.simplednn.utils.progressindicator.ProgressIndicatorBar

/**
 *
 */
abstract class TrainingHelper<ExampleType: Example>(
  open val neuralProcessor: NeuralProcessor,
  val optimizer: ParamsOptimizer,
  val lossCalculator: LossCalculator,
  val verbose: Boolean = false) {

  /**
   *
   */
  val statistics = TrainingStatistics()

  /**
   *
   */
  private var startTime: Long = 0

  /**
   *
   * @param trainingExamples training examples
   * @param validationExamples validation examples
   * @param validationHelper the helper for the validation
   * @param epochs max epoch
   * @param batchSize the size of each batch (default 1)
   * @param shuffler enable shuffle trainingExamples
   * @return the last avgLoss
   */
  fun train(trainingExamples: ArrayList<ExampleType>,
            validationExamples: ArrayList<ExampleType>,
            validationHelper: ValidationHelper<ExampleType>?,
            epochs: Int,
            batchSize: Int,
            shuffler: Shuffler?): Unit {

    require(batchSize > 0)

    this.statistics.reset()

    for (i in 0 until epochs) {

      this.logTrainStart(epochIndex = i)

      this.trainEpoch(trainingExamples = trainingExamples, batchSize = batchSize, shuffler = shuffler)

      this.logTrainEnd()

      if (validationHelper != null) {

        this.logValidateStart()

        this.statistics.lastAccuracy = validationHelper.validate(validationExamples)

        this.logValidateEnd()
      }
    }
  }

  /**
   *
   */
  fun trainEpoch(trainingExamples: ArrayList<ExampleType>,
                 batchSize: Int,
                 shuffler: Shuffler?): Unit {

    this.newEpoch()

    val progress = ProgressIndicatorBar(trainingExamples.size)

    for (exampleIndex in ExamplesIndices(trainingExamples.size, shuffler = shuffler)) {

      progress.tick()

      this.statistics.lastLoss = this.trainExample(example = trainingExamples[exampleIndex], batchSize = batchSize)
    }
  }

  /**
   * Train the network from an example and accumulate the errors of the parameters into the optimizer
   *
   * @param example the example used to train the network
   *
   * @return the loss of the output compared to the expected gold
   */
  fun trainExample(example: ExampleType, batchSize: Int = 1): Double {

    if (this.statistics.exampleCount % batchSize == 0) { // A new batch starts
      this.newBatch()
    }

    this.newExample() // !! must be called after this.newBatch() !!

    val loss = this.learnFromExample(example)

    this.optimizer.accumulate(this.neuralProcessor.getParamsErrors())

    if (this.statistics.exampleCount == batchSize) { // A batch is just ended
      this.optimizer.update()
    }

    return loss
  }

  /**
   * Learn from an example (forward + backward)
   *
   * @param example the example used to train the network
   *
   * @return the loss of the output respect to the gold
   */
  abstract protected fun learnFromExample(example: ExampleType): Double

  /**
   * Method to call every new epoch.
   * It increments the epochCount and sets the batchCount and the exampleCount to zero
   *
   * In turn it calls the same method into the `optimizer`
   */
  private fun newEpoch(): Unit {
    this.statistics.newEpoch()
    this.optimizer.newEpoch()
  }

  /**
   * Method to call every new batch.
   * It increments the batchCount and sets the exampleCount to zero
   *
   * In turn it calls the same method into the `optimizer`
   */
  private fun newBatch(): Unit {
    this.statistics.newBatch()
    this.optimizer.newBatch()
  }

  /**
   * Method to call every new example.
   * It increments the exampleCount
   *
   * In turn it calls the same method into the `optimizer`
   */
  private fun newExample(): Unit {
    this.statistics.newExample()
    this.optimizer.newExample()
  }

  /**
   *
   */
  private fun logTrainStart(epochIndex: Int): Unit {

    if (this.verbose) {

      this.startTime = System.currentTimeMillis()

      println("Epoch ${epochIndex + 1}")
    }
  }

  /**
   *
   */
  private fun logTrainEnd(): Unit {

    if (this.verbose) { // TODO: replace lastLoss with another more valuable value

      val elapsedTime = System.currentTimeMillis() - this.startTime

      println("[%d ms] Loss: %.10f".format(elapsedTime, this.statistics.lastLoss))
    }
  }

  /**
   *
   */
  private fun logValidateStart(): Unit {

    if (this.verbose) {

      this.startTime = System.currentTimeMillis()

      println("Start validation")
    }
  }

  /**
   *
   */
  private fun logValidateEnd(): Unit {

    if (this.verbose) { // TODO: replace lastLoss with another more valuable value

      val elapsedTime = System.currentTimeMillis() - this.startTime

      println("[%d ms] Accuracy: %.2f%%".format(elapsedTime, 100.0 * this.statistics.lastAccuracy))
    }
  }
}
