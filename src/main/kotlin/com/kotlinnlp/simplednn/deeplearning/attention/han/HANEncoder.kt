/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attention.han

import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsAccumulator
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import com.kotlinnlp.simplednn.deeplearning.attention.attentionnetwork.AttentionNetwork
import com.kotlinnlp.simplednn.deeplearning.attention.attentionnetwork.AttentionNetworksPool
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNEncoder
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNEncodersPool
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * Encoder based on Hierarchical Attention Networks.
 *
 * Reference:
 *  [Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, Eduard Hovy - Hierarchical Attention Networks for
 *  Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)
 *
 * @property model the parameters of the model of the networks
 * @property propagateToInput whether to propagate the errors to the input during the [backward]
 * @param biRNNDropout the probability of dropout for the BiRNNs (default 0.0)
 * @param attentionDropout the probability of attention dropout (default 0.0)
 * @param outputDropout the probability of output dropout (default 0.0)
 * @param mePropK the k factor of the 'meProp' algorithm to propagate from the k (in percentage) output nodes with
 *                the top errors of the transform layers (ignored if null, the default)
 * @property id an identification number useful to track a specific [HANEncoder]
 */
class HANEncoder<InputNDArrayType: NDArray<InputNDArrayType>>(
  val model: HAN,
  override val propagateToInput: Boolean,
  biRNNDropout: Double = 0.0,
  attentionDropout: Double = 0.0,
  outputDropout: Double = 0.0,
  private val mePropK: Double? = null,
  override val id: Int = 0
) : NeuralProcessor<
  HierarchyItem, // InputType
  DenseNDArray, // OutputType
  DenseNDArray, // ErrorsType
  HierarchyItem // InputErrorsType
  > {

  /**
   * An array containing pools of encoders ([BiRNNEncoder]s), one for each level of the HAN.
   */
  private val encodersPools: List<BiRNNEncodersPool<*>> = List(
    size = this.model.hierarchySize,
    init = { i ->
      if (this.model.isInputLevel((i)))
        BiRNNEncodersPool<InputNDArrayType>(
          network = this.model.biRNNs[i],
          rnnDropout = biRNNDropout,
          mergeDropout = biRNNDropout,
          propagateToInput = this.propagateToInput)
      else
        BiRNNEncodersPool<DenseNDArray>(
          network = this.model.biRNNs[i],
          rnnDropout = biRNNDropout,
          mergeDropout = biRNNDropout,
          propagateToInput = true)
    }
  )

  /**
   * An array containing pools of [AttentionNetwork]s, one for each level of the HAN.
   */
  private val attentionNetworksPools: List<AttentionNetworksPool<DenseNDArray>> = List(
    size = this.model.hierarchySize,
    init = { i ->
      AttentionNetworksPool<DenseNDArray>(
        model = this.model.attentionNetworksParams[i],
        inputType = LayerType.Input.Dense,
        propagateToInput = this.propagateToInput,
        dropout = attentionDropout)
    }
  )

  /**
   * Lists of encoders associated to each forwarded group, one per hierarchical level.
   */
  private val usedEncodersPerLevel: List<MutableList<BiRNNEncoder<*>>> = List(
    size = this.model.hierarchySize,
    init = { mutableListOf<BiRNNEncoder<*>>() }
  )

  /**
   * Lists of attention networks associated to each forwarded group, one per hierarchical level.
   */
  private val usedAttentionNetworksPerLevel: List<MutableList<AttentionNetwork<DenseNDArray>>> = List(
    size = this.model.hierarchySize,
    init = { mutableListOf<AttentionNetwork<DenseNDArray>>() }
  )

  /**
   * An array containing params errors accumulator for each BiRNN encoder.
   */
  private val encodersParamsErrorsAccumulators: List<ParamsErrorsAccumulator> =
    List(this.model.hierarchySize) { ParamsErrorsAccumulator() }

  /**
   * An array containing params errors accumulator for each [AttentionNetwork].
   */
  private val attentionNetworksParamsErrorsAccumulators: List<ParamsErrorsAccumulator> =
    List(this.model.hierarchySize) { ParamsErrorsAccumulator() }

  /**
   * The processor for the output Feedforward network (single layer).
   */
  private val outputProcessor: FeedforwardNeuralProcessor<DenseNDArray> =
    FeedforwardNeuralProcessor(model = this.model.outputNetwork, propagateToInput = true, dropout = outputDropout)

  /**
   * Forward a sequences hierarchy encoding the sequence of each level through a [BiRNNEncoder] and an
   * [AttentionNetwork].
   *
   * The output of the top level is classified using a single Feedforward Layer.
   *
   * @param input the sequences hierarchy of input
   *
   * @return the output [DenseNDArray]
   */
  override fun forward(input: HierarchyItem): DenseNDArray {

    this.resetUsedNetworks()

    val topOutput: DenseNDArray = this.forwardItem(item = input, levelIndex = 0)

    return this.outputProcessor.forward(topOutput)
  }

  /**
   * Propagate the errors from the output within the whole hierarchical structure, eventually until the input.
   *
   * @param outputErrors the errors of the output
   */
  override fun backward(outputErrors: DenseNDArray) {

    this.resetAccumulators()

    this.outputProcessor.backward(outputErrors)

    this.backwardHierarchicalGroup(
      outputErrors = this.outputProcessor.getInputErrors(copy = false),
      levelIndex = 0,
      groupIndex = 0)
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the input sequences, grouped with the same hierarchy as the given input
   */
  override fun getInputErrors(copy: Boolean): HierarchyItem =
    this.buildInputErrorsHierarchyItem(levelIndex = 0, groupIndex = 0, copy = copy)

  /**
   * The lowest level of the [HierarchyItem] contains [HierarchySequence]s containing only one [DenseNDArray] with the
   * importance score of the related input sequence.
   * Sum the scores of a group to obtain the score of the upper hierarchy level.
   * The sum of the scores of all the input sequences is equal to 1.0.
   *
   * @return the importance scores of the input sequences, grouped with the same hierarchy as the given input
   */
  fun getInputImportanceScores(): HierarchyItem {
    return this.buildImportanceScoreHierarchyItem(levelIndex = 0, groupIndex = 0, refScore = 1.0)
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the HAN parameters
   */
  override fun getParamsErrors(copy: Boolean): ParamsErrorsList {

    val paramsErrors = mutableListOf<ParamsErrorsList>()

    paramsErrors.add(List(
      size = this.model.hierarchySize,
      init = { i ->
        this.encodersParamsErrorsAccumulators[i].getParamsErrors(copy = copy)
      }).flatten())


    paramsErrors.add(List(
      size = this.model.hierarchySize,
      init = { i ->
        this.attentionNetworksParamsErrorsAccumulators[i].getParamsErrors(copy = copy)
      }).flatten())

    paramsErrors.add(this.outputProcessor.getParamsErrors(copy = copy))

    return paramsErrors.flatten()
  }

  /**
   * Apply the forward to the given [item] of the hierarchy dispatching it between a 'level' or a 'sequence'.
   *
   * @param item the item of the hierarchy to which to apply the forward
   * @param levelIndex the index of the hierarchical level of the [item]
   *
   * @return the output array
   */
  @Suppress("UNCHECKED_CAST")
  private fun forwardItem(item: HierarchyItem, levelIndex: Int): DenseNDArray {

    val inputSequence: List<*> = when (item) {
      is HierarchyGroup -> this.buildInputSequence(group = item, levelIndex = levelIndex)
      is HierarchySequence<*> -> item
      else -> throw RuntimeException("Invalid hierarchy item type")
    }

    val encoder: BiRNNEncoder<*> = this.encodersPools[levelIndex].getItem()
    val attentionNetwork: AttentionNetwork<DenseNDArray> = this.attentionNetworksPools[levelIndex].getItem()

    this.usedEncodersPerLevel[levelIndex].add(encoder)
    this.usedAttentionNetworksPerLevel[levelIndex].add(attentionNetwork)

    val isInputLevel: Boolean = this.model.isInputLevel(levelIndex)

    val encodedSequence: List<DenseNDArray> = if (isInputLevel) {
      (encoder as BiRNNEncoder<InputNDArrayType>).forward(inputSequence as List<InputNDArrayType>)
    } else {
      (encoder as BiRNNEncoder<DenseNDArray>).forward(inputSequence as List<DenseNDArray>)
    }

    return attentionNetwork.forward(encodedSequence)
  }

  /**
   * Build the input sequence of the given [group] forwarding its sub-levels .
   *
   * @param group a group of the hierarchy
   * @param levelIndex the index of the hierarchical level of the given [group]
   *
   * @return the input sequence for the given [group]
   */
  private fun buildInputSequence(group: HierarchyGroup, levelIndex: Int): List<DenseNDArray> =
    group.map { this.forwardItem(item = it, levelIndex = levelIndex + 1) }

  /**
   * Backward the hierarchy group at the given [levelIndex] of the hierarchy, with the given [groupIndex].
   *
   * @param outputErrors the errors of the output of the [AttentionNetwork] at the given [levelIndex]
   * @param levelIndex the index of the propagating level of the hierarchy
   * @param groupIndex the index of the propagating group of this level
   */
  private fun backwardHierarchicalGroup(outputErrors: DenseNDArray,
                                        levelIndex: Int,
                                        groupIndex: Int) {

    val isNotLastLevel = levelIndex < (this.model.hierarchySize - 1)

    this.backwardAttentionNetwork(
      outputErrors = outputErrors,
      levelIndex = levelIndex,
      groupIndex = groupIndex)

    this.backwardEncoder(
      levelIndex = levelIndex,
      groupIndex = groupIndex)

    if (isNotLastLevel) {
      this.usedEncodersPerLevel[levelIndex][groupIndex].getInputErrors().forEachIndexed { i, errors ->
        this.backwardHierarchicalGroup(
          outputErrors = errors,
          levelIndex = levelIndex + 1,
          groupIndex = i
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
    val paramsErrors = this.usedAttentionNetworksPerLevel[levelIndex][groupIndex].backward(outputErrors)

    accumulator.accumulate(paramsErrors)
  }

  /**
   * Backward of the [BiRNNEncoder] at the given [levelIndex] of the hierarchy.
   *
   * @param levelIndex the index of a level in the hierarchy
   * @param groupIndex the index of the propagating group of this level
   */
  private fun backwardEncoder(levelIndex: Int, groupIndex: Int) {

    val accumulator = this.encodersParamsErrorsAccumulators[levelIndex]
    val encoder: BiRNNEncoder<*> = this.usedEncodersPerLevel[levelIndex][groupIndex]

    encoder.backward(this.usedAttentionNetworksPerLevel[levelIndex][groupIndex].getInputErrors())

    accumulator.accumulate(encoder.getParamsErrors(copy = false))
  }

  /**
   * Reset all the params errors accumulators.
   */
  private fun resetAccumulators() {

    this.encodersParamsErrorsAccumulators.forEach { it.clear() }
    this.attentionNetworksParamsErrorsAccumulators.forEach { it.clear() }
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
   * Build the [HierarchyItem] of the given [levelIndex] of the hierarchy, related to the given [groupIndex] group.
   * If the level is the last the returned [HierarchyItem] is a [HierarchySequence] containing the input errors of the
   * given [groupIndex] group.
   *
   * @param levelIndex the index of a level of the hierarchy
   * @param groupIndex the index of a group in this level
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return a [HierarchyItem] containing input errors on the lowest level
   */
  private fun buildInputErrorsHierarchyItem(levelIndex: Int, groupIndex: Int, copy: Boolean): HierarchyItem {

    return if (levelIndex == (this.model.hierarchySize - 1))
      HierarchySequence(
        *this.usedEncodersPerLevel[levelIndex][groupIndex].getInputErrors(copy = copy).toTypedArray()
      )
    else
      HierarchyGroup(*Array(
        size = this.usedEncodersPerLevel[levelIndex][groupIndex].getInputErrors(copy = false).size,
        init = { i -> this.buildInputErrorsHierarchyItem(levelIndex = levelIndex + 1, groupIndex = i, copy = copy) }
      ))
  }

  /**
   * Build the [HierarchyItem] of the given [levelIndex] of the hierarchy, related to the given [groupIndex] group.
   * If the level is the last the returned [HierarchyItem] is a [HierarchySequence] containing the importance scores of
   * the given [groupIndex] group.
   *
   * @param levelIndex the index of a level of the hierarchy
   * @param groupIndex the index of a group in this level
   * @param refScore the reference score on which to base the distribution of the scores of the given group
   *
   * @return a [HierarchyItem] containing the importance score on the lowest level
   */
  private fun buildImportanceScoreHierarchyItem(levelIndex: Int, groupIndex: Int, refScore: Double): HierarchyItem {

    val importanceScores: DenseNDArray =
      this.usedAttentionNetworksPerLevel[levelIndex][groupIndex].getImportanceScore(copy = false)

    return if (levelIndex == (this.model.hierarchySize - 1))
      HierarchySequence(
        importanceScores.prod(refScore)
      )

    else
      HierarchyGroup(*Array(
        size = importanceScores.length,
        init = { i ->
          this.buildImportanceScoreHierarchyItem(
            levelIndex = levelIndex + 1,
            groupIndex = i,
            refScore = importanceScores[i])
        }
      ))
  }
}
