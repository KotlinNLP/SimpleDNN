/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package traininghelpers.training

import com.kotlinnlp.simplednn.core.optimizer.GenericParamsOptimizer
import traininghelpers.validation.ValidationHelper
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar
import com.kotlinnlp.utils.*
import utils.Example


/**
 * The TrainingHelper.
 *
 * @property optimizer the optimizer (can be null)
 * @property verbose whether to print training details
 */
abstract class TrainingHelper<ExampleType: Example>(
  open val optimizer: GenericParamsOptimizer?,
  val verbose: Boolean = false) {

  /**
   * The statistics of training (accuracy, loss, etc..).
   */
  val statistics = TrainingStatistics()

  /**
   * When timing started.
   */
  private var startTime: Long = 0

  /**
   * Train over the specified number of [epochs], grouping examples in
   * batches of the given [batchSize] and shuffling them with the given [shuffler] before each epoch.
   * If [validationHelper] is not null, the NeuralNetwork is tested over the given [validationExamples] after each
   * epoch.
   *
   * @param trainingExamples training examples
   * @param epochs number of epochs
   * @param batchSize the size of each batch (default 1)
   * @param validationExamples validation examples (default null)
   * @param validationHelper the helper for the validation (default null)
   * @param shuffler the [Shuffler] to shuffle [trainingExamples] before each epoch (default null)
   */
  fun train(trainingExamples: List<ExampleType>,
            epochs: Int,
            batchSize: Int,
            validationExamples: List<ExampleType>? = null,
            validationHelper: ValidationHelper<ExampleType>? = null,
            shuffler: Shuffler? = null) {

    require(batchSize > 0)

    this.statistics.reset()

    for (i in 0 until epochs) {

      this.logTrainingStart(epochIndex = i)

      this.trainEpoch(trainingExamples = trainingExamples, batchSize = batchSize, shuffler = shuffler)

      this.logTrainingEnd()

      if (validationHelper != null) {
        require(validationExamples != null)

        this.logValidationStart(validationExamples!!.size)

        this.statistics.lastAccuracy = validationHelper.validate(validationExamples)

        this.logValidationEnd()
      }
    }
  }

  /**
   * Train grouping examples in batches of the given [batchSize] and
   * shuffling them with the given [shuffler] before training.
   *
   * @param trainingExamples training examples
   * @param batchSize the size of each batch (default 1)
   * @param shuffler the [Shuffler] to shuffle [trainingExamples] before training (default null)
   */
  private fun trainEpoch(trainingExamples: List<ExampleType>, batchSize: Int, shuffler: Shuffler? = null) {

    this.newEpoch()

    val progress = ProgressIndicatorBar(trainingExamples.size)

    for (exampleIndex in ExamplesIndices(trainingExamples.size, shuffler = shuffler)) {

      progress.tick()

      this.statistics.lastLoss = this.trainExample(example = trainingExamples[exampleIndex], batchSize = batchSize)
    }
  }

  /**
   * Train the network with the given [example] and accumulate the errors of the parameters into the [optimizer].
   *
   * @param example the example used to train the network
   *
   * @return the loss of the output compared to the expected gold
   */
  private fun trainExample(example: ExampleType, batchSize: Int = 1): Double {

    if (this.statistics.exampleCount % batchSize == 0) { // A new batch starts
      this.newBatch()
    }

    this.newExample() // !! must be called after this.newBatch() !!

    val loss = this.learnFromExample(example)

    this.accumulateParamsErrors(batchSize)

    if (this.statistics.exampleCount == batchSize) { // a batch is just ended
      this.optimizer?.update()
    }

    return loss
  }

  /**
   * Learn from an example (forward + backward).
   *
   * @param example the example used to train the network
   *
   * @return the loss of the output respect to the gold
   */
  protected abstract fun learnFromExample(example: ExampleType): Double

  /**
   * Accumulate the params errors resulting from [learnFromExample].
   *
   * @param batchSize the size of each batch
   */
  protected abstract fun accumulateParamsErrors(batchSize: Int)

  /**
   * Method to call every new epoch.
   * It increments the epochCount and sets the batchCount and the exampleCount to zero
   *
   * In turn it calls the same method into the `optimizer`
   */
  private fun newEpoch() {
    this.statistics.newEpoch()
    this.optimizer?.newEpoch()
  }

  /**
   * Method to call every new batch.
   * It increments the batchCount and sets the exampleCount to zero
   *
   * In turn it calls the same method into the `optimizer`
   */
  private fun newBatch() {
    this.statistics.newBatch()
    this.optimizer?.newBatch()
  }

  /**
   * Method to call every new example.
   * It increments the exampleCount
   *
   * In turn it calls the same method into the `optimizer`
   */
  private fun newExample() {
    this.statistics.newExample()
    this.optimizer?.newExample()
  }

  /**
   * Log when training starts.
   */
  private fun logTrainingStart(epochIndex: Int) {

    if (this.verbose) {

      this.startTiming()

      println("\nEpoch ${epochIndex + 1}")
    }
  }

  /**
   * Log when training ends.
   */
  private fun logTrainingEnd() {

    if (this.verbose) { // TODO: replace lastLoss with another more valuable value
      println("Elapsed time: %s".format(this.formatElapsedTime()))
      println("Loss: %.5f".format(100.0 * this.statistics.lastLoss))
    }
  }

  /**
   * Log when validation starts.
   */
  private fun logValidationStart(validationExamplesSize: Int) {

    if (this.verbose) {

      this.startTiming()

      println("Validate on $validationExamplesSize examples")
    }
  }

  /**
   * Log when validation ends.
   */
  private fun logValidationEnd() {

    if (this.verbose) {
      println("Elapsed time: %s".format(this.formatElapsedTime()))
      println("Accuracy: %.2f%%".format(100.0 * this.statistics.lastAccuracy))
    }
  }

  /**
   * Start registering time.
   */
  private fun startTiming() {
    this.startTime = System.currentTimeMillis()
  }

  /**
   * @return the formatted string with elapsed time in seconds and minutes.
   */
  private fun formatElapsedTime(): String {

    val elapsedTime = System.currentTimeMillis() - this.startTime
    val elapsedSecs = elapsedTime / 1000.0

    return "%.3f s (%.1f min)".format(elapsedSecs, elapsedSecs / 60.0)
  }
}
