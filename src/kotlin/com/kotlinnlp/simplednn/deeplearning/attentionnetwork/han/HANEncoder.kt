/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attentionnetwork.han

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.LayerUnit
import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerStructure
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.AttentionNetwork
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNEncoder
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * Hierarchical Attention Networks.
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
  private val encoders = Array(
    size = this.model.hierarchySize,
    init = { i -> BiRNNEncoder<DenseNDArray>(this.model.biRNNs[i]) }
  )

  /**
   * An array containing the [AttentionNetwork]s, one for each level of the HAN.
   */
  private val attentionNetworks = Array(
    size = this.model.hierarchySize,
    init = { i ->
      AttentionNetwork<DenseNDArray>(
        model = this.model.attentionNetworksParams[i],
        inputType = LayerType.Input.Dense,
        dropout = this.dropout)
    }
  )

  /**
   *
   */
  private val outputLayer = FeedforwardLayerStructure(
    inputArray = AugmentedArray(DenseNDArrayFactory.emptyArray(shape = Shape(this.model.biRNNs.last().hiddenSize))),
    outputArray = LayerUnit<DenseNDArray>(size = this.model.outputSize),
    params = this.model.outputLayerParams
  )

  /**
   * Forward a sequences hierarchy encoding the sequence of each level through a [BiRNNEncoder] and an
   * [AttentionNetwork].
   *
   * The output of the top level is classified using a single Feedforward Layer with a Softmax activation.
   *
   * @param sequencesHierarchy the sequences hierarchy of input
   * @param useDropout whether to apply the dropout to generate the attention arrays
   *
   * @return the output [DenseNDArray]
   */
  fun forward(sequencesHierarchy: HierarchyItem, useDropout: Boolean = false): DenseNDArray {

    val outputArray: DenseNDArray = this.forwardItem(item = sequencesHierarchy, levelIndex = 0, useDropout = useDropout)

    this.outputLayer.setInput(outputArray)
    this.outputLayer.forward(useDropout = useDropout)

    return this.outputLayer.outputArray.values
  }

  /**
   *
   */
  fun backward(outputErrors: DenseNDArray) {
    TODO("not implemented")
  }

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
}
