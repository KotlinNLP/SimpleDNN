/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralprocessor.embeddingsprocessor

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.merge.concat.ConcatLayer
import com.kotlinnlp.simplednn.core.layers.models.merge.concat.ConcatLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.merge.concat.ConcatLayersPool
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsAccumulator
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The NeuralProcessor that acts on an embeddings map.
 *
 * The particularity of this variant of [EmbeddingsProcessor] is that a shared context vector is concatenated
 * to each embedding.
 *
 * @param embeddingsMap the embeddings map
 * @param contextVector the context vector to concatenate to each embedding
 * @param useDropout whether to apply the dropout during the forward
 */
class EmbeddingsProcessorWithContext<T>(
  embeddingsMap: EmbeddingsMap<T>,
  private val contextVector: ParamsArray,
  useDropout: Boolean
) : EmbeddingsProcessor<T>(
  embeddingsMap = embeddingsMap,
  useDropout = useDropout
){

  companion object {

    /**
     * The index of the embeddings in the input arrays of the concat layer.
     */
    private const val embdIndex = 0

    /**
     * The index of the context vector in the input arrays of the concat layer.
     */
    private const val cntxIndex = 1
  }

  /**
   * Pool of concat layers.
   */
  private val concatLayersPool = ConcatLayersPool<DenseNDArray>(
    params = ConcatLayerParameters(inputsSize = listOf(embeddingsMap.size, this.contextVector.values.length)),
    inputType = LayerType.Input.Dense
  )

  /**
   * The concat layers used during the last forward.
   */
  private val concatLayers = mutableListOf<ConcatLayer<DenseNDArray>>()

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

    this.initConcatLayers(input.size)

    return super.forward(input).mapIndexed { i, embedding ->

      this.concatLayers[i].let {

        it.setInput(embdIndex, embedding)
        it.setInput(cntxIndex, this.contextVector.values)
        it.forward()
        it.outputArray.values
      }
    }
  }

  /**
   * The Backward.
   *
   * @param outputErrors the output errors
   */
  override fun backward(outputErrors: List<DenseNDArray>) {

    this.backwardConcatLayers(outputErrors)

    super.backward(this.concatLayers.map { it.inputArrays[embdIndex].errors })

    this.accumulateContextVectorErrors(this.concatLayers.map { it.inputArrays[cntxIndex].errors })
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
   * Initialize the [concatLayers].
   *
   * @param size the number of concat layers to initialize
   */
  private fun initConcatLayers(size: Int) {

    this.concatLayersPool.releaseAll()
    this.concatLayers.addAll((0 until size).map { this.concatLayersPool.getItem() } )
  }

  /**
   * Perform the backward of the [concatLayers].
   *
   * @param outputErrors the errors to propagate
   */
  private fun backwardConcatLayers(outputErrors: List<DenseNDArray>) {

    outputErrors.forEachIndexed { i, errors ->

      this.concatLayers[i].let {
        it.setErrors(errors)
        it.backward(propagateToInput = true)
      }
    }
  }

  /**
   * Accumulate the errors of the context vector.
   *
   * @param outputErrors the errors to accumulate
   */
  private fun accumulateContextVectorErrors(outputErrors: List<DenseNDArray>) {

    this.contextErrorsAccumulator.clear()
    this.contextErrorsAccumulator.accumulate(params = contextVector, errors = outputErrors)
    this.contextErrorsAccumulator.averageErrors()
  }
}
