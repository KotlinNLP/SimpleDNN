/* Copyright 2019-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.helpers

/**
 * A counter of values used during the training.
 */
internal data class Counter(
  var epoch: Int = 0,
  var batch: Int = 0,
  var exampleCount: Int = 0,
  var bestAccuracy: Double = 0.0
) {

  /**
   * Reset all the counters.
   */
  fun reset() {
    this.epoch = 0
    this.batch = 0
    this.exampleCount = 0
    this.bestAccuracy = 0.0
  }

  /**
   * Method to call every new epoch.
   * It increments the epochs counter and it sets the batches and the examples counters to zero.
   */
  fun newEpoch() {
    this.epoch += 1
    this.batch = 0
    this.exampleCount = 0
  }

  /**
   * Method to call every new batch.
   * It increments the batches counter and it sets the examples counter to zero.
   */
  fun newBatch() {
    this.batch += 1
    this.exampleCount = 0
  }

  /**
   * Method to call every new example.
   * It increments the examples counter.
   */
  fun newExample() {
    this.exampleCount += 1
  }
}

