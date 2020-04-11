/* Copyright 2019-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.helpers

import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar
import com.kotlinnlp.utils.*

/**
 * The trainer of a neural model.
 *
 * @param modelFilename the name of the file in which to save the serialized model
 * @param optimizers the parameters optimizers
 * @param examples the training examples
 * @param epochs the number of training epochs
 * @param batchSize the size of each batch (default 1)
 * @param evaluator the helper for the evaluation (default null)
 * @param shuffler shuffle the examples before each epoch, converting the sequence to a list
 * @param verbose whether to print info about the training progress and timing (default = true)
 */
abstract class Trainer<ExampleType : Any>(
  protected val modelFilename: String,
  protected val optimizers: List<ParamsOptimizer>,
  private val examples: Iterable<ExampleType>,
  private val epochs: Int,
  protected val batchSize: Int = 1,
  private val evaluator: Evaluator<ExampleType, *>? = null,
  private val shuffler: Shuffler? = Shuffler(),
  protected val verbose: Boolean = true
) {

  /**
   * An iterator of shuffled examples.
   *
   * @param examples the examples list
   */
  private inner class ShuffledExamplesIterator(private val examples: List<ExampleType>) : Iterator<ExampleType> {

    /**
     * The iterator of shuffled examples indices.
     */
    private val indicesIterator = ShuffledIndices(this.examples.size, shuffler = shuffler!!).iterator()

    override fun next(): ExampleType = this.examples[indicesIterator.next()]

    override fun hasNext(): Boolean = indicesIterator.hasNext()
  }

  /**
   * Counter of values used during the training (accuracy, loss, etc..).
   */
  private val counter = Counter()

  /**
   * A timer to track the elapsed time.
   */
  private val timer = Timer()

  /**
   * The best accuracy reached.
   */
  val bestAccuracy get() = this.counter.bestAccuracy

  /**
   * Check requirements.
   */
  init {
    require(this.epochs > 0)
    require(this.batchSize > 0)
  }

  /**
   * Train the model over the specified number of epochs, grouping the examples in batches, eventually shuffled before.
   * If the [evaluator] is not null, the neural model is tested with validation examples after each epoch.
   */
  fun train() {

    this.counter.reset()

    (0 until this.epochs).forEach { i ->

      this.logTrainingStart(epochIndex = i)

      this.newEpoch()
      this.trainEpoch()

      this.logTrainingEnd()

      this.validateAndSaveModel()
    }
  }

  /**
   * Learn from an example (forward + backward).
   *
   * @param example an example to train the model with
   */
  protected abstract fun learnFromExample(example: ExampleType)

  /**
   * Accumulate the errors of the model resulting after the call of [learnFromExample].
   */
  protected abstract fun accumulateErrors()

  /**
   * Dump the model to file.
   */
  protected abstract fun dumpModel()

  /**
   * Train the model over an epoch, grouping examples in batches, shuffling them before with the given shuffler.
   */
  protected open fun trainEpoch() {

    val examplesIterator: Iterator<ExampleType> = this.buildExamplesIterator()
    val progress: ProgressIndicatorBar? =
      if (this.examples is Collection<*>) ProgressIndicatorBar(this.examples.size) else null

    while (examplesIterator.hasNext()) {

      if (this.verbose) progress?.tick()

      if (this.counter.exampleCount % this.batchSize == 0) // A new batch starts
        this.newBatch()

      this.newExample() // !! must be called after newBatch() !!

      this.trainExample(examplesIterator.next())
    }
  }

  /**
   * @return an iterator of examples
   */
  private fun buildExamplesIterator(): Iterator<ExampleType> =
    if (this.shuffler != null)
      ShuffledExamplesIterator(if (this.examples is List<ExampleType>) this.examples else this.examples.toList())
    else
      this.examples.iterator()

  /**
   * Train the neural model with a given example and accumulate the errors into the [optimizers].
   *
   * @param example an example to train the model with
   */
  private fun trainExample(example: ExampleType) {

    this.learnFromExample(example)

    this.accumulateErrors()

    if (this.counter.exampleCount == this.batchSize) { // a batch is just ended
      this.optimizers.forEach { it.update() }
    }
  }

  /**
   * Validate the model and save it to file if a new best accuracy has been reached.
   * If the [evaluator] is null then the model is saved automatically.
   */
  private fun validateAndSaveModel() {

    var bestModel = true

    if (this.evaluator != null) {

      this.logValidationStart()

      val stats: Statistics = this.evaluator.evaluate()

      this.logValidationEnd(stats)

      if (stats.accuracy > this.counter.bestAccuracy)
        this.counter.bestAccuracy = stats.accuracy
      else
        bestModel = false
    }

    if (bestModel) {

      this.dumpModel()

      if (this.verbose)
        println("Model saved to \"${this.modelFilename}\"")
    }
  }

  /**
   * Method to call every new epoch.
   * In turn it calls the same method of the [optimizers].
   */
  protected fun newEpoch() {
    this.counter.newEpoch()
    this.optimizers.forEach { it.newEpoch() }
  }

  /**
   * Method to call every new batch.
   * In turn it calls the same method of the [optimizers].
   */
  protected fun newBatch() {
    this.counter.newBatch()
    this.optimizers.forEach { it.newBatch() }
  }

  /**
   * Method to call every new example.
   * In turn it calls the same method of the [optimizers].
   */
  protected fun newExample() {
    this.counter.newExample()
    this.optimizers.forEach { it.newExample() }
  }

  /**
   * Log when training starts.
   */
  private fun logTrainingStart(epochIndex: Int) {

    if (this.verbose) {

      this.timer.reset()

      println("\nEpoch ${epochIndex + 1} of ${this.epochs}")
      println("\nStart training...")
    }
  }

  /**
   * Log when training ends.
   */
  private fun logTrainingEnd() {

    if (this.verbose) {
      println("Elapsed time: %s".format(this.timer.formatElapsedTime()))
    }
  }

  /**
   * Log when validation starts.
   */
  private fun logValidationStart() {

    if (this.verbose) {

      val evaluationExamples: Iterable<ExampleType> = this.evaluator!!.examples

      this.timer.reset()

      if (evaluationExamples is Collection<*>)
        println("\nValidate on ${evaluationExamples.size} examples...")
      else
        println("\nValidate model...")
    }
  }

  /**
   * Log when validation ends.
   *
   * @param stats the evaluation statistics
   */
  private fun logValidationEnd(stats: Statistics) {

    if (this.verbose) {

      println("Elapsed time: %s".format(this.timer.formatElapsedTime()))

      println("\nStatistics:")
      println(stats)

      if (stats.accuracy > this.counter.bestAccuracy)
        println("\nNEW BEST ACCURACY!")
    }
  }
}
