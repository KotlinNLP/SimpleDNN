/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attentionnetwork.attentionlayer

import com.kotlinnlp.simplednn.core.optimizer.Optimizer
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.utils.scheduling.BatchScheduling
import com.kotlinnlp.simplednn.utils.scheduling.EpochScheduling
import com.kotlinnlp.simplednn.utils.scheduling.ExampleScheduling

/**
 * The optimizer of the Attention Layer.
 *
 * @property params the attention layer parameters to optimize
 * @property updateMethod the [UpdateMethod] for the optimization (e.g. ADAM, AdaGrad, ...)
 */
class AttentionLayerOptimizer(val params: AttentionLayerParameters, val updateMethod: UpdateMethod) : Optimizer {

  /**
   * A support structure to store the errors of the context vector.
   */
  private val contextVectorErrors: DenseNDArray = this.params.contextVector.values.zerosLike()

  /**
   * The counter of the amount of errors accumulated.
   */
  private var count: Int = 0

  /**
   * Accumulate the parameters errors contained into the [errors].
   *
   * @param errors the errors of the Attention Layer parameters
   */
  fun accumulateErrors(errors: AttentionLayerParameters) {

    this.contextVectorErrors.assignSum(errors.contextVector.values)
    this.count += 1
  }

  /**
   * Update the parameters.
   */
  override fun update() {

    this.contextVectorErrors.assignDiv(this.count.toDouble()) // average errors
    this.updateMethod.update(this.params.contextVector, this.contextVectorErrors)

    this.contextVectorErrors.zeros()
    this.count = 0
  }

  /**
   * Method to call every new epoch.
   * In turn it calls the same method into the `updateMethod`.
   */
  override fun newEpoch() {
    if (this.updateMethod is EpochScheduling) {
      this.updateMethod.newEpoch()
    }
  }

  /**
   * Method to call every new batch.
   * In turn it calls the same method into the `updateMethod`.
   */
  override fun newBatch() {
    if (this.updateMethod is BatchScheduling) {
      this.updateMethod.newBatch()
    }
  }

  /**
   * Method to call every new example.
   * In turn it calls the same method into the `updateMethod`.
   */
  override fun newExample() {
    if (this.updateMethod is ExampleScheduling) {
      this.updateMethod.newExample()
    }
  }
}
