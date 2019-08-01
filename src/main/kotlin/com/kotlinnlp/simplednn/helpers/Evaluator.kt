/* Copyright 2019-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.helpers

import com.kotlinnlp.simplednn.core.optimizer.IterableParams
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar

/**
 * A helper which evaluates a neural model with a list of examples.
 *
 * @param examples a list of examples to validate
 * @param params the parameters of the neural model to evaluate
 * @param verbose whether to print info about the validation progress (default = true)
 */
abstract class Evaluator<ExampleType : Any>(
  internal val examples: List<ExampleType>,
  internal val params: IterableParams<*>,
  private val verbose: Boolean = true
) {

  /**
   * The evaluation statistics.
   */
  protected abstract val stats: Statistics

  /**
   * Evaluate examples.
   *
   * @return the validation statistics
   */
  fun evaluate(): Statistics {

    val progress = ProgressIndicatorBar(this.examples.size)

    this.stats.reset()

    this.examples.forEach {

      this.evaluate(it)

      if (this.verbose) progress.tick()
    }

    return this.stats
  }

  /**
   * Evaluate the model with a single example.
   *
   * @param example the example to validate the model with
   */
  protected abstract fun evaluate(example: ExampleType)
}
