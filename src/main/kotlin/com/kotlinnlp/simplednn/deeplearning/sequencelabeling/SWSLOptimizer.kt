/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.sequencelabeling

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adagrad.AdaGradMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.optimizer.Optimizer
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.core.optimizer.ScheduledUpdater
import com.kotlinnlp.simplednn.embeddings.EmbeddingsOptimizer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The SWSLOptimizer is the optimizer of the [SWSLNetwork].
 *
 * @property network the [SWSLNetwork] to optimize
 * @property paramsUpdateMethod the [UpdateMethod] to optimize the network params (default ADAM)
 * @property embeddingsUpdateMethod the [UpdateMethod] to optimize the embeddings (default AdaGrad)
 */
class SWSLOptimizer(
  private val network: SWSLNetwork,
  private val paramsUpdateMethod: UpdateMethod<*> = ADAMMethod(stepSize = 0.001),
  private val embeddingsUpdateMethod: UpdateMethod<*> = AdaGradMethod(learningRate = 0.1)
) : ScheduledUpdater {

  /**
   * The [Optimizer] used to optimize the network
   */
  private val classifierOptimizer = ParamsOptimizer(this.network.classifier.model, this.paramsUpdateMethod)

  /**
   * The [Optimizer] used to optimize the labels embeddings.
   *
   * The update method is AdaGrad with learning-rate = 0.1
   *
   */
  private val labelEmbeddingsOptimizer = EmbeddingsOptimizer(
    embeddingsMap = this.network.labelsEmbeddings,
    updateMethod = embeddingsUpdateMethod)

  /**
   * Update the params (network params and labels embeddings) using the accumulated errors.
   * After the update the errors are cleared.
   */
  override fun update() {
    this.classifierOptimizer.update()
    this.labelEmbeddingsOptimizer.update()
  }

  /**
   * Method to call every new epoch.
   * In turn it calls the same method into the `updateMethod`
   */
  override fun newEpoch() {
    this.classifierOptimizer.newEpoch()
    this.labelEmbeddingsOptimizer.newEpoch()
  }

  /**
   * Method to call every new batch.
   * In turn it calls the same method into the `updateMethod`
   */
  override fun newBatch() {
    this.classifierOptimizer.newBatch()
    this.labelEmbeddingsOptimizer.newBatch()
  }

  /**
   * Method to call every new example.
   * In turn it calls the same method into the `updateMethod`
   */
  override fun newExample() {
    this.classifierOptimizer.newExample()
    this.labelEmbeddingsOptimizer.newExample()
  }

  /**
   * Accumulate network params errors.
   *
   * @param errors the params errors to accumulate
   */
  fun accumulateParamsErrors(errors: NetworkParameters) {
    this.classifierOptimizer.accumulate(errors)
  }

  /**
   * Accumulate embeddings errors.
   *
   * @param embeddingId the id of an embedding
   * @param errors the errors of this embedding
   */
  fun accumulateLabelEmbeddingErrors(embeddingId: Int, errors: DenseNDArray) {
    this.labelEmbeddingsOptimizer.accumulate(embeddingKey = embeddingId, errors = errors)
  }
}
