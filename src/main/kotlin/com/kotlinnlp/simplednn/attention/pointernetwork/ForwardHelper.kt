/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.attention.pointernetwork

import com.kotlinnlp.simplednn.core.mergelayers.MergeLayer
import com.kotlinnlp.simplednn.core.attentionlayer.AttentionMechanismStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The forward helper of the [PointerNetwork].
 *
 * @property network the attentive recurrent network of this helper
 */
class ForwardHelper(
  private val network: PointerNetwork
) {

  /**
   * @param vector the vector that modulates a content-based attention mechanism over the input sequence
   *
   * @return an array that contains the importance score for each element of the input sequence
   */
  fun forward(vector: DenseNDArray): DenseNDArray {

    if (this.network.firstState) this.initForward()

    val attentionArrays: List<DenseNDArray> = this.buildAttentionSequence(vector)

    return this.buildAttentionStructure(attentionArrays).forwardImportanceScore()
  }

  /**
   * @param attentionArrays the attention arrays
   *
   * @return the attention structure
   */
  private fun buildAttentionStructure(attentionArrays: List<DenseNDArray>): AttentionMechanismStructure {

    this.network.usedAttentionStructures.add(
      AttentionMechanismStructure(attentionArrays, params = this.network.model.attentionParams))

    return this.network.usedAttentionStructures.last()
  }

  /**
   * @param vector the vector that modulates a content-based attention mechanism over the input sequence
   *
   * @return the sequence of attention arrays
   */
  private fun buildAttentionSequence(vector: DenseNDArray): List<DenseNDArray> {

    val transformLayers = this.getTransformLayers(size = this.network.inputSequence.size)

    return ArrayList(transformLayers.zip(this.network.inputSequence).map { (layer, element) ->

      layer.setInput(0, element)
      layer.setInput(1, vector)
      layer.forward()

      layer.outputArray.values
    })
  }

  /**
   * Get an available group of transform layers, adding it into the usedTransformLayers list.
   *
   * @param size the number of transform layer to build
   *
   * @return an available transform layer
   */
  private fun getTransformLayers(size: Int): List<MergeLayer<DenseNDArray>> {

    this.network.usedTransformLayers.add(
      List(size = size, init = { this.network.transformLayersPool.getItem() })
    )

    return this.network.usedTransformLayers.last()
  }

  /**
   * Initialize the structures used during the forward.
   */
  private fun initForward() {

    this.network.transformLayersPool.releaseAll()
    this.network.usedTransformLayers.clear()
    this.network.usedAttentionStructures.clear()
  }
}
