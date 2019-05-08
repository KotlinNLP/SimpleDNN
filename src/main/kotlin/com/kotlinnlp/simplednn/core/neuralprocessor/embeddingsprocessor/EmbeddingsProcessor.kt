/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralprocessor.embeddingsprocessor

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsAccumulator
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The NeuralProcessor that acts on an embeddings map.
 *
 * @param embeddingsMap the embeddings map
 * @param dropout the dropout to mask items of the [embeddingsMap]
 * @param useDropout whether to apply the dropout during the forward
 */
open class EmbeddingsProcessor<T>(
  private val embeddingsMap: EmbeddingsMap<T>,
  private val dropout: Double = 0.0,
  override val useDropout: Boolean
) : NeuralProcessor<
  List<T>, // InputType
  List<DenseNDArray>, // OutputType
  List<DenseNDArray>, // ErrorsType
  NeuralProcessor.NoInputErrors // InputErrorsType
  > {

  /**
   * Whether to propagate the errors to the input during the backward (not supported)
   */
  override val propagateToInput: Boolean = false

  /**
   * The id for the pool (not supported).
   */
  override val id: Int = 0

  /**
   * List of embeddings used during the last forward.
   */
  private var usedEmbeddings = listOf<ParamsArray>()

  /**
   * List of embeddings errors resulting from the last backward.
   */
  private val errorsAccumulator by lazy { ParamsErrorsAccumulator() }

  /**
   * The Forward.
   *
   * @param input the input
   *
   * @return the result of the forward
   */
  override fun forward(input: List<T>): List<DenseNDArray> {

    val dropoutValue: Double = if (this.useDropout) this.dropout else 0.0

    this.usedEmbeddings = input.map { this.embeddingsMap.get(it, dropoutValue) }

    return this.usedEmbeddings.map { it.values } // TODO: copy?
  }

  /**
   * The Backward.
   *
   * @param outputErrors the output errors
   */
  override fun backward(outputErrors: List<DenseNDArray>) {

    require(outputErrors.size == this.usedEmbeddings.size) {
      "Number of errors (%d) does not reflect the number of used embeddings (%d)".format(
        outputErrors.size, this.usedEmbeddings.size)
    }

    this.errorsAccumulator.clear()

    this.usedEmbeddings.zip(outputErrors).forEach { (embedding, errors) ->

      this.errorsAccumulator.accumulate(embedding, errors)
    }

    this.errorsAccumulator.averageErrors()
  }

  /**
   * Return the input errors of the last backward.
   * Before calling this method make sure that [propagateToInput] is enabled.
   *
   * @param copy whether to return by value or by reference (default true)
   *
   * @return the input errors
   */
  override fun getInputErrors(copy: Boolean) = NeuralProcessor.NoInputErrors

  /**
   * Return the params errors of the last backward.
   *
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference (default true)
   *
   * @return the parameters errors
   */
  override fun getParamsErrors(copy: Boolean) = this.errorsAccumulator.getParamsErrors(copy = copy)
}