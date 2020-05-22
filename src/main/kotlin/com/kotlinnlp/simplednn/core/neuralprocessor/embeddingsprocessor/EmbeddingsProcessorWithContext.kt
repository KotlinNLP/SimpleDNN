/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralprocessor.embeddingsprocessor

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.simplednn.core.layers.LayerInterface
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.simplednn.core.neuralprocessor.batchfeedforward.BatchFeedforwardProcessor
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsAccumulator
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * Extension of the [EmbeddingsProcessor] with a context vector that is concatenated to each embedding.
 *
 * The [dropout] has no effect on the [contextVector].
 *
 * @param embeddingsMap an embeddings map
 * @param contextVector the context vector concatenated to each embedding
 * @param dropout the probability to get the unknown embedding (default 0.0)
 */
class EmbeddingsProcessorWithContext<T>(
  embeddingsMap: EmbeddingsMap<T>,
  private val contextVector: ParamsArray,
  dropout: Double = 0.0
) : EmbeddingsProcessor<T>(
  embeddingsMap = embeddingsMap,
  dropout = dropout
) {

  /**
   * The processor that concatenates the [contextVector] to each embedding.
   */
  private val concatProcessor = BatchFeedforwardProcessor<DenseNDArray>(
    model = StackedLayersParameters(
      LayerInterface(sizes = listOf(embeddingsMap.size, contextVector.values.length), type = LayerType.Input.Dense),
      LayerInterface(sizes = listOf(), connectionType = LayerType.Connection.Concat)),
    propagateToInput = true)

  /**
   * Accumulator of the errors of the [contextVector].
   */
  private val contextErrorsAccumulator by lazy { ParamsErrorsAccumulator() }

  /**
   * The Forward.
   *
   * @param input the input
   *
   * @return the result of the forward
   */
  override fun forward(input: List<T>): List<DenseNDArray> {

    val embeddings: List<DenseNDArray> = super.forward(input)

    return this.concatProcessor.forward(embeddings.map { listOf(it, this.contextVector.values) }.toTypedArray())
  }

  /**
   * The Backward.
   *
   * @param outputErrors the output errors
   */
  override fun backward(outputErrors: List<DenseNDArray>) {

    val concatErrors: List<List<DenseNDArray>> = this.concatProcessor.let {
      it.backward(outputErrors)
      it.getInputsErrors(copy = false)
    }

    super.backward(concatErrors.map { it.first() })

    this.accumulateContextVectorErrors(concatErrors.map { it.last() })
  }

  /**
   * Return the params errors of the last backward.
   *
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference (default true)
   *
   * @return the parameters errors
   */
  override fun getParamsErrors(copy: Boolean) =
    super.getParamsErrors(copy) + this.contextErrorsAccumulator.getParamsErrors(copy)

  /**
   * Accumulate the errors of the context vector.
   *
   * @param outputErrors the errors to accumulate
   */
  private fun accumulateContextVectorErrors(outputErrors: List<DenseNDArray>) {

    this.contextErrorsAccumulator.clear()
    this.contextErrorsAccumulator.accumulate(params = this.contextVector, errors = outputErrors)
  }
}
