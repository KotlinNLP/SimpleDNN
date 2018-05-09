/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attentionnetwork.attentionmechanism

import com.kotlinnlp.simplednn.core.optimizer.Optimizer
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The optimizer of the [AttentionParameters].
 *
 * @property params the attention parameters to optimize
 * @property updateMethod the [UpdateMethod] for the optimization (e.g. ADAM, AdaGrad, ...)
 */
class AttentionOptimizer(
  val params: AttentionParameters,
  updateMethod: UpdateMethod<*>
) : Optimizer(updateMethod) {

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
   * @param errors the errors of the Attention parameters
   */
  fun accumulateErrors(errors: AttentionParameters) {

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
}
