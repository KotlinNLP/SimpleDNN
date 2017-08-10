/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.helpers.training.utils

/**
 *
 */
data class TrainingStatistics(
  var epochCount: Int = 0,
  var batchCount: Int = 0,
  var exampleCount: Int = 0,
  var lastLoss: Double = 0.0,
  var lastAccuracy: Double = 0.0) {

  /**
   *
   */
  fun reset(): Unit {
    this.epochCount = 0
    this.batchCount = 0
    this.exampleCount = 0
    this.lastLoss  = 0.0
    this.lastAccuracy = 0.0
  }

  /**
   * Method to call every new epoch.
   *
   * It increments the epochCount and sets the batchCount and the exampleCount to zero
   *
   */
  fun newEpoch(): Unit {
    this.epochCount += 1
    this.batchCount = 0
    this.exampleCount = 0
  }

  /**
   * Method to call every new batch.

   * It increments the batchCount and sets the exampleCount to zero
   *
   */
  fun newBatch(): Unit {
    this.batchCount += 1
    this.exampleCount = 0
  }

  /**
   * Method to call every new example.
   *
   * It increments the exampleCount
   *
   */
  fun newExample(): Unit {
    this.exampleCount += 1
  }
}

