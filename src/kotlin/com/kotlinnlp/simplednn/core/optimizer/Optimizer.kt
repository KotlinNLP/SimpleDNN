/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.optimizer

import com.kotlinnlp.simplednn.utils.scheduling.BatchScheduling
import com.kotlinnlp.simplednn.utils.scheduling.EpochScheduling
import com.kotlinnlp.simplednn.utils.scheduling.ExampleScheduling

/**
 *
 */
interface Optimizer : ExampleScheduling, BatchScheduling, EpochScheduling {
  /**
   *
   */
  fun update()

  /**
   * Method to call every new epoch.
   */
  override fun newEpoch()

  /**
   * Method to call every new batch.
   */
  override fun newBatch()

  /**
   * Method to call every new example.
   */
  override fun newExample()
}
