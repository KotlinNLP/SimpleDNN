/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.optimizer

import com.kotlinnlp.simplednn.core.functionalities.gradientclipping.GradientClipping
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.utils.scheduling.BatchScheduling
import com.kotlinnlp.simplednn.utils.scheduling.EpochScheduling
import com.kotlinnlp.simplednn.utils.scheduling.ExampleScheduling

/**
 * The optimizer of neural parameters.
 *
 * @param updateMethod the update method helper (Learning Rate, ADAM, AdaGrad, ...)
 * @param gradientClipping the gradient clipper (default null)
 */
class ParamsOptimizer(
  private val updateMethod: UpdateMethod<*>,
  private val gradientClipping: GradientClipping? = null
) : ParamsErrorsAccumulator(), ScheduledUpdater {

  /**
   * Calculate the errors average, update the parameters.
   */
  override fun update() {

    if (this.isNotEmpty) {
      
      this.averageErrors()
      this.clipGradients()
      this.updateParams()
      this.clear()
    }
  }

  /**
   * Method to call every new epoch.
   * In turn it calls the same method into the `updateMethod`
   */
  override fun newEpoch() {

    if (this.updateMethod is EpochScheduling) {
      this.updateMethod.newEpoch()
    }
  }

  /**
   * Method to call every new batch.
   * In turn it calls the same method into the `updateMethod`
   */
  override fun newBatch() {

    if (this.updateMethod is BatchScheduling) {
      this.updateMethod.newBatch()
    }
  }

  /**
   * Method to call every new example.
   * In turn it calls the same method into the `updateMethod`
   */
  override fun newExample() {

    if (this.updateMethod is ExampleScheduling) {
      this.updateMethod.newExample()
    }
  }

  /**
   * Update the parameters in respect of the errors using the update method helper (Learning Rate, ADAM, AdaGrad, ...).
   */
  private fun updateParams() {

    this.getParamsErrors(copy = false).forEach { errors ->

      this.updateMethod.update(array = errors.refParams, errors = errors.values)
    }
  }

  /**
   * Perform the gradient clipping.
   */
  private fun clipGradients() {

    this.gradientClipping?.clip(this.getParamsErrors(copy = false))
  }
}
