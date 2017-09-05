package com.kotlinnlp.simplednn.core.optimizer

import com.kotlinnlp.simplednn.utils.scheduling.BatchScheduling
import com.kotlinnlp.simplednn.utils.scheduling.EpochScheduling
import com.kotlinnlp.simplednn.utils.scheduling.ExampleScheduling

/**
 * Defines a module which optimizes the parameters of neural elements and is scheduled by training events.
 */
interface ScheduledUpdater : ExampleScheduling, BatchScheduling, EpochScheduling {

  /**
   * Update the parameters of the neural elements associated to this [ScheduledUpdater].
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
