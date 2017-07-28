/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attentionnetwork.han

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.AttentionNetwork
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.AttentionNetworkParameters
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.AttentionNetworkParamsErrorsAccumulator
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNEncoder
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNParamsErrorsAccumulator
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * Encoder based on Hierarchical Attention Networks.
 *
 *   Reference:
 *   [Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, Eduard Hovy -
 *   Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)
 *
 * @property model the parameters of the model of the networks
 * @property dropout the probability of dropout (default 0.0) when generating the attention arrays for the Attention
 *                   Layers. If applying it, the usual value is 0.5 (better 0.25 if it's the first layer).
 */
class HANEncoder(val model: HAN, val dropout: Double = 0.0) {

  /**
   * An array containing the encoders ([BiRNNEncoder]s), one for each level of the HAN.
   */
  private val encoders: Array<BiRNNEncoder<DenseNDArray>> = Array(
    size = this.model.hierarchySize,
    init = { i -> BiRNNEncoder<DenseNDArray>(this.model.biRNNs[i]) }
  )

  /**
   * An array containing params errors accumulator for each BiRNN encoder.
   */
  private val encodersParamsErrorsAccumulators: Array<BiRNNParamsErrorsAccumulator> = Array(
    size = this.model.hierarchySize,
    init = { i -> BiRNNParamsErrorsAccumulator(this.model.biRNNs[i]) }
  )

  /**
   * An array containing the [AttentionNetwork]s, one for each level of the HAN.
   */
  private val attentionNetworks: Array<AttentionNetwork<DenseNDArray>> = Array(
    size = this.model.hierarchySize,
    init = { i ->
      AttentionNetwork<DenseNDArray>(
        model = this.model.attentionNetworksParams[i],
        inputType = LayerType.Input.Dense,
        dropout = this.dropout)
    }
  )

  /**
   * An array containing params errors accumulator for each [AttentionNetwork].
   */
  private val attentionNetworksParamsErrorsAccumulators: Array<AttentionNetworkParamsErrorsAccumulator> = Array(
    size = this.model.hierarchySize,
    init = { i ->
      AttentionNetworkParamsErrorsAccumulator(
        inputSize = this.attentionNetworks[i].model.inputSize,
        attentionSize = this.attentionNetworks[i].model.attentionSize,
        sparseInput = false)
    }
  )

  /**
   * An array containing structures to save the params errors of each [AttentionNetwork] during the backward.
   */
  private val attentionNetworksParamsErrors: Array<AttentionNetworkParameters> = Array(
    size = this.model.hierarchySize,
    init = { i-> AttentionNetworkParameters(
      inputSize = this.attentionNetworks[i].model.inputSize,
      attentionSize = this.attentionNetworks[i].model.attentionSize,
      sparseInput = false)
    }
  )

  /**
   * The processor for the output Feedforward network (single layer).
   */
  private val outputProcessor = FeedforwardNeuralProcessor<DenseNDArray>(this.model.outputNetwork)

  /**
   * Forward a sequences hierarchy encoding the sequence of each level through a [BiRNNEncoder] and an
   * [AttentionNetwork].
   *
   * The output of the top level is classified using a single Feedforward Layer.
   *
   * @param sequencesHierarchy the sequences hierarchy of input
   * @param useDropout whether to apply the dropout to generate the attention arrays
   *
   * @return the output [DenseNDArray]
   */
  fun forward(sequencesHierarchy: HierarchyItem, useDropout: Boolean = false): DenseNDArray {

    val topOutput: DenseNDArray = this.forwardItem(item = sequencesHierarchy, levelIndex = 0, useDropout = useDropout)

    this.outputProcessor.forward(featuresArray = topOutput, useDropout = useDropout)

    return this.outputProcessor.getOutput()
  }

  /**
   * Propagate the errors from the output within the whole hierarchical structure, eventually until the input.
   *
   * @param outputErrors the errors of the output
   * @param propagateToInput whether to propagate the output errors to the input or not
   */
  fun backward(outputErrors: DenseNDArray, propagateToInput: Boolean) {

    this.resetAccumulators()

    this.outputProcessor.backward(outputErrors = outputErrors, propagateToInput = true)

    this.backwardHierarchicalLevel(
      outputErrors = this.outputProcessor.getInputErrors(copy = false),
      levelIndex = 0,
      propagateToInput = propagateToInput
    )

    this.averageAccumulatedErrors()
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the input sequences, grouped with the same hierarchy as the given input
   */
  fun getInputSequenceErrors(copy: Boolean = true): HierarchyItem {
    TODO("not implemented")
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the HAN parameters
   */
  fun getParamsErrors(copy: Boolean = true) = HANParameters(
    biRNNs = arrayListOf(*Array(
      size = this.model.hierarchySize,
      init = { i ->
        val paramsErrors = this.encodersParamsErrorsAccumulators[i].getParamsErrors()
        if (copy) paramsErrors.clone() else paramsErrors
      }
    )),
    attentionNetworks = arrayListOf(*Array(
      size = this.model.hierarchySize,
      init = { i ->
        val paramsErrors = this.attentionNetworksParamsErrorsAccumulators[i].getParamsErrors()
        if (copy) paramsErrors.clone() else paramsErrors
      }
    )),
    outputNetwork = this.outputProcessor.getParamsErrors(copy = copy)
  )

  /**
   * Apply the forward to the given [item] of the hierarchy dispatching it between a 'level' or a 'sequence'.
   *
   * @param item the item of the hierarchy to which to apply the forward
   * @param levelIndex the index of a level in the hierarchy
   * @param useDropout whether to apply the dropout to generate the attention arrays
   *
   * @return the output array
   */
  private fun forwardItem(item: HierarchyItem, levelIndex: Int, useDropout: Boolean): DenseNDArray {

    val inputSequence = when (item) {
      is HierarchyLevel -> this.buildInputSequence(level = item, levelIndex = levelIndex, useDropout = useDropout)
      is HierarchySequence -> item.toTypedArray()
      else -> throw RuntimeException("Invalid hierarchy item type")
    }

    val encodedSequence = this.encoders[levelIndex].encode(inputSequence)

    return this.attentionNetworks[levelIndex].forward(
      inputSequence = arrayListOf(*Array(
        size = encodedSequence.size,
        init = { i -> AugmentedArray(encodedSequence[i])}
      )),
      useDropout = useDropout)
  }

  /**
   * Build the input sequence of the given [level] forwarding its sub-levels .
   *
   * @param level a level of the hierarchy
   * @param levelIndex the index of the given [level]
   * @param useDropout whether to apply the dropout to generate the attention arrays
   *
   * @return the input sequence for the given [level]
   */
  private fun buildInputSequence(level: HierarchyLevel,
                                 levelIndex: Int,
                                 useDropout: Boolean): Array<DenseNDArray> {

    return Array(
      size = level.size,
      init = { i ->
        this.forwardItem(item = level[i], levelIndex = levelIndex + 1, useDropout = useDropout)
      }
    )
  }

  /**
   * Backward of the whole level at the given [levelIndex] of the hierarchy.
   *
   * @param outputErrors the errors of the output of the [AttentionNetwork] at the given [levelIndex]
   * @param levelIndex the index of a level in the hierarchy
   * @param propagateToInput whether to propagate the errors to the input or not
   */
  private fun backwardHierarchicalLevel(outputErrors: DenseNDArray, levelIndex: Int, propagateToInput: Boolean) {

    val isNotLastLevel = levelIndex < (this.model.hierarchySize - 1)

    this.backwardAttentionNetwork(outputErrors = outputErrors, levelIndex = levelIndex)
    this.backwardEncoder(levelIndex = levelIndex, propagateToInput = isNotLastLevel || propagateToInput)

    if (isNotLastLevel) {
      this.encoders[levelIndex].getInputSequenceErrors().forEach {
        this.backwardHierarchicalLevel(
          outputErrors = it,
          levelIndex = levelIndex + 1,
          propagateToInput = propagateToInput
        )
      }
    }
  }

  /**
   * Backward of the [AttentionNetwork] at the given [levelIndex] of the hierarchy.
   *
   * @param outputErrors the errors of the output of the [AttentionNetwork] at the given [levelIndex]
   * @param levelIndex the index of a level in the hierarchy
   */
  private fun backwardAttentionNetwork(outputErrors: DenseNDArray, levelIndex: Int) {

    val accumulator = this.attentionNetworksParamsErrorsAccumulators[levelIndex]
    val paramsErrors = this.attentionNetworksParamsErrors[levelIndex]

    this.attentionNetworks[levelIndex].backward(
      outputErrors = outputErrors,
      paramsErrors = paramsErrors,
      propagateToInput = true)

    accumulator.accumulate(paramsErrors)
  }

  /**
   * Backward of the [BiRNNEncoder] at the given [levelIndex] of the hierarchy.
   *
   * @param levelIndex the index of a level in the hierarchy
   * @param propagateToInput whether to propagate the errors to the input or not
   */
  private fun backwardEncoder(levelIndex: Int, propagateToInput: Boolean) {

    val accumulator = this.encodersParamsErrorsAccumulators[levelIndex]
    val encoder: BiRNNEncoder<DenseNDArray> = this.encoders[levelIndex]

    encoder.backward(
      outputErrorsSequence = this.attentionNetworks[levelIndex].getInputErrors(),
      propagateToInput = propagateToInput)

    accumulator.accumulate(encoder.getParamsErrors(copy = false))
  }

  /**
   * Set the average of all the params errors accumulated by each accumulator.
   */
  private fun averageAccumulatedErrors() {

    this.encodersParamsErrorsAccumulators.forEach {
      it.averageErrors()
    }

    this.attentionNetworksParamsErrorsAccumulators.forEach {
      it.averageErrors()
    }
  }

  /**
   * Reset all the params errors accumulators.
   */
  private fun resetAccumulators() {

    this.encodersParamsErrorsAccumulators.forEach {
      it.reset()
    }

    this.attentionNetworksParamsErrorsAccumulators.forEach {
      it.reset()
    }
  }
}
