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
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsAccumulator
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.AttentionNetwork
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.AttentionNetworkParameters
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.AttentionNetworksPool
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNEncoder
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNEncodersPool
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * Encoder based on Hierarchical Attention Networks.
 *
 *   Reference:
 *   [Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, Eduard Hovy -
 *   Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)
 *
 * @property model the parameters of the model of the networks
 */
class HANEncoder(val model: HAN) {

  /**
   * An array containing pools of encoders ([BiRNNEncoder]s), one for each level of the HAN.
   */
  private val encodersPools: Array<BiRNNEncodersPool<DenseNDArray>> = Array(
    size = this.model.hierarchySize,
    init = { i -> BiRNNEncodersPool<DenseNDArray>(this.model.biRNNs[i]) }
  )

  /**
   * An array containing pools of [AttentionNetwork]s, one for each level of the HAN.
   */
  private val attentionNetworksPools: Array<AttentionNetworksPool<DenseNDArray>> = Array(
    size = this.model.hierarchySize,
    init = { i ->
      AttentionNetworksPool<DenseNDArray>(
        model = this.model.attentionNetworksParams[i],
        inputType = LayerType.Input.Dense)
    }
  )

  /**
   * Lists of encoders associated to each forwarded group, one per hierarchical level.
   */
  private val usedEncodersPerLevel: List<ArrayList<BiRNNEncoder<DenseNDArray>>> = List(
    size = this.model.hierarchySize,
    init = { arrayListOf<BiRNNEncoder<DenseNDArray>>() }
  )

  /**
   * Lists of attention networks associated to each forwarded group, one per hierarchical level.
   */
  private val usedAttentionNetworksPerLevel: List<ArrayList<AttentionNetwork<DenseNDArray>>> = List(
    size = this.model.hierarchySize,
    init = { arrayListOf<AttentionNetwork<DenseNDArray>>() }
  )

  /**
   * An array containing params errors accumulator for each BiRNN encoder.
   */
  private val encodersParamsErrorsAccumulators: Array<ParamsErrorsAccumulator<BiRNNParameters>> = Array(
    size = this.model.hierarchySize,
    init = { ParamsErrorsAccumulator<BiRNNParameters>() }
  )

  /**
   * An array containing params errors accumulator for each [AttentionNetwork].
   */
  private val attentionNetworksParamsErrorsAccumulators = Array(
    size = this.model.hierarchySize,
    init = { ParamsErrorsAccumulator<AttentionNetworkParameters>() }
  )

  /**
   * An array containing structures to save the params errors of each [AttentionNetwork] during the backward.
   */
  private val attentionNetworksParamsErrors: Array<AttentionNetworkParameters> = Array(
    size = this.model.hierarchySize,
    init = { i-> AttentionNetworkParameters(
      inputSize = this.attentionNetworksPools[i].model.inputSize,
      attentionSize = this.attentionNetworksPools[i].model.attentionSize,
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
   * @param useDropout whether to apply the dropout
   *
   * @return the output [DenseNDArray]
   */
  fun forward(sequencesHierarchy: HierarchyItem, useDropout: Boolean = false): DenseNDArray {

    this.resetUsedNetworks()

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

    this.backwardHierarchicalGroup(
      outputErrors = this.outputProcessor.getInputErrors(copy = false),
      levelIndex = 0,
      groupIndex = 0,
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

    return this.buildInputErrorsHierarchyItem(levelIndex = 0, groupIndex = 0, copy = copy)
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the HAN parameters
   */
  fun getParamsErrors(copy: Boolean = true) = HANParameters(
    biRNNs = Array(
      size = this.model.hierarchySize,
      init = { i ->
        val paramsErrors = this.encodersParamsErrorsAccumulators[i].getParamsErrors()
        if (copy) paramsErrors.copy() else paramsErrors
      }
    ),
    attentionNetworks = Array(
      size = this.model.hierarchySize,
      init = { i ->
        val paramsErrors = this.attentionNetworksParamsErrorsAccumulators[i].getParamsErrors()
        if (copy) paramsErrors.copy() else paramsErrors
      }
    ),
    outputNetwork = this.outputProcessor.getParamsErrors(copy = copy)
  )

  /**
   * Apply the forward to the given [item] of the hierarchy dispatching it between a 'level' or a 'sequence'.
   *
   * @param item the item of the hierarchy to which to apply the forward
   * @param levelIndex the index of the hierarchical level of the [item]
   * @param useDropout whether to apply the dropout to generate the attention arrays
   *
   * @return the output array
   */
  private fun forwardItem(item: HierarchyItem, levelIndex: Int, useDropout: Boolean): DenseNDArray {

    val inputSequence = when (item) {
      is HierarchyGroup -> this.buildInputSequence(group = item, levelIndex = levelIndex, useDropout = useDropout)
      is HierarchySequence -> item.toTypedArray()
      else -> throw RuntimeException("Invalid hierarchy item type")
    }

    val encoder: BiRNNEncoder<DenseNDArray> = this.encodersPools[levelIndex].getItem()
    val attentionNetwork: AttentionNetwork<DenseNDArray> = this.attentionNetworksPools[levelIndex].getItem()

    this.usedEncodersPerLevel[levelIndex].add(encoder)
    this.usedAttentionNetworksPerLevel[levelIndex].add(attentionNetwork)

    val encodedSequence = encoder.encode(inputSequence, useDropout = useDropout)

    return attentionNetwork.forward(
      inputSequence = arrayListOf(*Array(
        size = encodedSequence.size,
        init = { i -> AugmentedArray(encodedSequence[i])}
      )),
      useDropout = useDropout)
  }

  /**
   * Build the input sequence of the given [group] forwarding its sub-levels .
   *
   * @param group a group of the hierarchy
   * @param levelIndex the index of the hierarchical level of the given [group]
   * @param useDropout whether to apply the dropout to generate the attention arrays
   *
   * @return the input sequence for the given [group]
   */
  private fun buildInputSequence(group: HierarchyGroup,
                                 levelIndex: Int,
                                 useDropout: Boolean): Array<DenseNDArray> {

    return Array(
      size = group.size,
      init = { i ->
        this.forwardItem(item = group[i], levelIndex = levelIndex + 1, useDropout = useDropout)
      }
    )
  }

  /**
   * Backward the hierarchy group at the given [levelIndex] of the hierarchy, with the given [groupIndex].
   *
   * @param outputErrors the errors of the output of the [AttentionNetwork] at the given [levelIndex]
   * @param levelIndex the index of the propagating level of the hierarchy
   * @param groupIndex the index of the propagating group of this level
   * @param propagateToInput whether to propagate the errors to the input or not
   */
  private fun backwardHierarchicalGroup(outputErrors: DenseNDArray,
                                        levelIndex: Int,
                                        groupIndex: Int,
                                        propagateToInput: Boolean) {

    val isNotLastLevel = levelIndex < (this.model.hierarchySize - 1)

    this.backwardAttentionNetwork(
      outputErrors = outputErrors,
      levelIndex = levelIndex,
      groupIndex = groupIndex)

    this.backwardEncoder(
      levelIndex = levelIndex,
      groupIndex = groupIndex,
      propagateToInput = isNotLastLevel || propagateToInput)

    if (isNotLastLevel) {
      this.usedEncodersPerLevel[levelIndex][groupIndex].getInputSequenceErrors().forEachIndexed { i, errors ->
        this.backwardHierarchicalGroup(
          outputErrors = errors,
          levelIndex = levelIndex + 1,
          groupIndex = i,
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
   * @param groupIndex the index of the propagating group of this level
   */
  private fun backwardAttentionNetwork(outputErrors: DenseNDArray, levelIndex: Int, groupIndex: Int) {

    val accumulator = this.attentionNetworksParamsErrorsAccumulators[levelIndex]
    val paramsErrors = this.attentionNetworksParamsErrors[levelIndex]

    this.usedAttentionNetworksPerLevel[levelIndex][groupIndex].backward(
      outputErrors = outputErrors,
      paramsErrors = paramsErrors,
      propagateToInput = true)

    accumulator.accumulate(paramsErrors)
  }

  /**
   * Backward of the [BiRNNEncoder] at the given [levelIndex] of the hierarchy.
   *
   * @param levelIndex the index of a level in the hierarchy
   * @param groupIndex the index of the propagating group of this level
   * @param propagateToInput whether to propagate the errors to the input or not
   */
  private fun backwardEncoder(levelIndex: Int, groupIndex: Int, propagateToInput: Boolean) {

    val accumulator = this.encodersParamsErrorsAccumulators[levelIndex]
    val encoder: BiRNNEncoder<DenseNDArray> = this.usedEncodersPerLevel[levelIndex][groupIndex]

    encoder.backward(
      outputErrorsSequence = this.usedAttentionNetworksPerLevel[levelIndex][groupIndex].getInputErrors(),
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

  /**
   * Set all sub-networks as not used.
   */
  private fun resetUsedNetworks() {

    this.usedEncodersPerLevel.forEach { it.clear() }
    this.usedAttentionNetworksPerLevel.forEach { it.clear() }

    this.encodersPools.forEach { it.releaseAll() }
    this.attentionNetworksPools.forEach { it.releaseAll() }
  }

  /**
   * Build the [HierarchyItem] of the given [levelIndex] of the hierarchy, related to the group with the given
   * [groupIndex].
   * If the level is the last the returned [HierarchyItem] is a [HierarchySequence] containing the input errors of the
   * group with index [groupIndex].
   *
   * @param levelIndex the index of a level of the hierarchy
   * @param groupIndex the index of a group in this level
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   */
  private fun buildInputErrorsHierarchyItem(levelIndex: Int, groupIndex: Int, copy: Boolean): HierarchyItem {

    val subGroupSize: Int = this.usedEncodersPerLevel[levelIndex][groupIndex].getInputSequenceErrors(copy = false).size

    return if (levelIndex == (this.model.hierarchySize - 1))
      HierarchySequence(*this.usedEncodersPerLevel[levelIndex][groupIndex].getInputSequenceErrors(copy = copy))
    else
      HierarchyGroup(*Array(
        size = subGroupSize,
        init = { i -> this.buildInputErrorsHierarchyItem(levelIndex = levelIndex + 1, groupIndex = i, copy = copy) }
      ))
  }
}
