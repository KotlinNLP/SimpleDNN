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
import com.kotlinnlp.simplednn.deeplearning.embeddings.EmbeddingsOptimizer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The SWSLOptimizer is the optimizer of the [SWSLNetwork].
 *
 * @property network the SWSLNetwork to optimize
 * @property updateMethod the [UpdateMethod] used to optimize the network params errors (default ADAM)
 */
class SWSLOptimizer(
  private val network: SWSLNetwork,
  val updateMethod: UpdateMethod = ADAMMethod(stepSize = 0.001)
) : Optimizer {

  /**
   * The [Optimizer] used to optimize the network
   */
  private val classifierOptimizer = ParamsOptimizer(this.network.classifier, this.updateMethod)

  /**
   * The [Optimizer] used to optimize the labels embeddings.
   *
   * The update method is AdaGrad with learning-rate = 0.1
   *
   */
  private val labelEmbeddingsOptimizer = EmbeddingsOptimizer(
    embeddingsContainer = this.network.labelsEmbeddings,
    updateMethod = AdaGradMethod(learningRate = 0.1))

  /**
   * Accumulate the embeddings errors
   *
   * @param embeddingIndex the index of an embedding
   * @param errors the errors of this embedding
   */
  fun accumulateLabelEmbeddingErrors(embeddingIndex: Int, errors: DenseNDArray) {
    this.labelEmbeddingsOptimizer.accumulateErrors(embeddingIndex = embeddingIndex, errors = errors)
  }

  /**
   * Method to call every new epoch.
   */
  override fun newEpoch() {
    this.classifierOptimizer.newEpoch()
    this.labelEmbeddingsOptimizer.newEpoch()
  }

  /**
   * Method to call every new batch.
   */
  override fun newBatch() {
    this.classifierOptimizer.newBatch()
    this.labelEmbeddingsOptimizer.newBatch()
  }

  /**
   * Method to call every new example.
   */
  override fun newExample() {
    this.classifierOptimizer.newExample()
    this.labelEmbeddingsOptimizer.newExample()
  }

  /**
   * Update the params (network params and labels embeddings) using the accumulated errors.
   * After the update the errors are cleared.
   */
  override fun update(): Unit {
    this.classifierOptimizer.update()
    this.labelEmbeddingsOptimizer.update()
  }

  /**
   * Accumulate the network params errors
   *
   * @param errors the params errors to accumulate
   */
  fun accumulateErrors(errors: NetworkParameters) {
    this.classifierOptimizer.accumulate(errors)
  }
}
