/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.embeddings

import com.kotlinnlp.simplednn.core.optimizer.Optimizer
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.utils.scheduling.BatchScheduling
import com.kotlinnlp.simplednn.utils.scheduling.EpochScheduling
import com.kotlinnlp.simplednn.utils.scheduling.ExampleScheduling

/**
 * The EmbeddingsOptimizer is the optimizer of the EmbeddingsContainer
 *
 * @property embeddingsContainer embeddings container to optimize
 * @property updateMethod the [UpdateMethod] for the optimization (e.g. ADAM, AdaGrad, ...)
 */
class EmbeddingsOptimizer(
  val embeddingsContainer: EmbeddingsContainer,
  val updateMethod: UpdateMethod) : Optimizer {

  /**
   * Support structure to store errors
   *
   * @property errors the [DenseNDArray] which contains the accumulated errors
   * @property count how many times the errors has been accumulated
   */
  private data class EmbeddingsErrors(val errors: DenseNDArray, var count: Int)

  /**
   * Map an embeddings index with its errors
   */
  private val embeddingsErrorsAccumulator = mutableMapOf<Int, EmbeddingsErrors>()

  /**
   * Accumulate errors of the [embeddingIndex] embedding
   *
   * @param embeddingIndex index of the embedding on which to accumulate the [errors]
   * @param errors errors to accumulate
   */
  fun accumulateErrors(embeddingIndex: Int, errors: DenseNDArray) {

    val embeddingsErrorsAccumulator: EmbeddingsErrors? = this.embeddingsErrorsAccumulator[embeddingIndex]

    if (embeddingsErrorsAccumulator == null){
      this.embeddingsErrorsAccumulator[embeddingIndex] = EmbeddingsErrors(errors = errors.copy(), count = 1)
    } else {
      embeddingsErrorsAccumulator.errors.assignSum(errors)
      embeddingsErrorsAccumulator.count += 1
    }
  }

  /**
   * Update the embeddings
   */
  override fun update() {

    for ((embeddingIndex, embeddingsErrors) in this.embeddingsErrorsAccumulator) {
      embeddingsErrors.errors.assignDiv(embeddingsErrors.count.toDouble()) // average errors
      this.updateMethod.update(this.embeddingsContainer.lookupTable[embeddingIndex].array, embeddingsErrors.errors)
    }

    this.embeddingsErrorsAccumulator.clear()
  }

  /**
   * Method to call every new epoch.
   *
   * In turn it calls the same method into the `updateMethod`
   */
  override fun newEpoch() {
    if (this.updateMethod is EpochScheduling) {
      this.updateMethod.newEpoch()
    }
  }

  /**
   * Method to call every new batch.
   *
   * In turn it calls the same method into the `updateMethod`
   */
  override fun newBatch() {
    if (this.updateMethod is BatchScheduling) {
      this.updateMethod.newBatch()
    }
  }

  /**
   * Method to call every new example.
   *
   * In turn it calls the same method into the `updateMethod`
   */
  override fun newExample() {
    if (this.updateMethod is ExampleScheduling) {
      this.updateMethod.newExample()
    }
  }
}
