/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attention.pointernetwork

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.models.attention.AttentionMechanismLayer
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The forward helper of the [PointerNetworkProcessor].
 *
 * @property networkProcessor the attentive recurrent network of this helper
 */
class ForwardHelper(private val networkProcessor: PointerNetworkProcessor) {

  /**
   * @param context the vector that modulates a content-based attention mechanism over the input sequence
   *
   * @return an array that contains the importance score for each element of the input sequence
   */
  fun forward(context: DenseNDArray): DenseNDArray {

    if (this.networkProcessor.firstState) this.initForward()

    val attentionArrays: List<DenseNDArray> = this.buildAttentionSequence(context)

    val attentionMechanism = this.buildAttentionMechanism(attentionArrays)

    attentionMechanism.forward()

    return attentionMechanism.outputArray.values
  }

  /**
   * @param attentionArrays the attention arrays
   *
   * @return an attention mechanisms
   */
  private fun buildAttentionMechanism(attentionArrays: List<DenseNDArray>): AttentionMechanismLayer =
    this.networkProcessor.usedAttentionMechanisms.addAndReturn(
      AttentionMechanismLayer(
        inputArrays = attentionArrays.map { AugmentedArray(it) },
        params = this.networkProcessor.model.attentionParams,
        activation = this.networkProcessor.model.activation))

  /**
   * @param context the vector that modulates a content-based attention mechanism over the input sequence
   *
   * @return the sequence of attention arrays
   */
  private fun buildAttentionSequence(context: DenseNDArray): List<DenseNDArray> {

    val mergeProcessors = this.getMergeProcessors(size = this.networkProcessor.inputSequence.size)

    return mergeProcessors.zip(this.networkProcessor.inputSequence).map { (processor, inputArray) ->
      processor.forward(featuresList = listOf(inputArray, context))
    }
  }

  /**
   * Add a new list of available merge processors into the usedMergeProcessors list and return it.
   *
   * @param size the number of merge processors to build
   *
   * @return a list of available merge processors
   */
  private fun getMergeProcessors(size: Int): List<FeedforwardNeuralProcessor<DenseNDArray>> =
    this.networkProcessor.usedMergeProcessors.addAndReturn(List(size = size, init = {
      this.networkProcessor.mergeProcessorsPool.getItem()
    }))

  /**
   * Initialize the structures used during the forward.
   */
  private fun initForward() {

    this.networkProcessor.mergeProcessorsPool.releaseAll()
    this.networkProcessor.usedMergeProcessors.clear()
    this.networkProcessor.usedAttentionMechanisms.clear()
  }

  /**
   * @param element the element to add to the list
   *
   * @return the given [element]
   */
  private fun <E>MutableList<E>.addAndReturn(element: E): E {
    this.add(element)
    return element
  }
}
