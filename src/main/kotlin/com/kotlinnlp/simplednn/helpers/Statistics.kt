/* Copyright 2019-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.helpers

import com.kotlinnlp.utils.stats.MetricCounter
import kotlin.properties.Delegates

/**
 * Validation statistics.
 */
abstract class Statistics {

  /**
   * Simple statistics with a simple metric of accuracy.
   */
  class Simple : Statistics() {

    /**
     * A metric counter.
     */
    val metric = MetricCounter()

    /**
     * Reset the metrics.
     */
    override fun reset() {

      this.accuracy = 0.0

      this.metric.reset()
    }

    override fun toString(): String = "Accuracy: %.2f%%".format(100.0 * this.accuracy)
  }

  /**
   * The overall accuracy of the model validated, in the range [0.0, 1.0].
   */
  var accuracy: Double by Delegates.observable(0.0) { _, _, newValue -> require(newValue in 0.0 .. 1.0) }

  /**
   * Reset the metrics.
   */
  abstract fun reset()
}
