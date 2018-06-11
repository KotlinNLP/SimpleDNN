/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attention.pointernetwork

import com.kotlinnlp.simplednn.core.layers.models.merge.MergeLayer
import com.kotlinnlp.simplednn.core.attention.AttentionMechanism
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The forward helper of the [PointerNetworkProcessor].
 *
 * @property networkProcessor the attentive recurrent network of this helper
 */
class ForwardHelper(private val networkProcessor: PointerNetworkProcessor) {

  /**
   * @param vector the vector that modulates a content-based attention mechanism over the input sequence
   *
   * @return an array that contains the importance score for each element of the input sequence
   */
  fun forward(vector: DenseNDArray): DenseNDArray {

    if (this.networkProcessor.firstState) this.initForward()

    val attentionArrays: List<DenseNDArray> = this.buildAttentionSequence(vector)

    return this.buildAttentionMechanism(attentionArrays).forwardImportanceScore()
  }

  /**
   * @param attentionArrays the attention arrays
   *
   * @return an attention mechanisms
   */
  private fun buildAttentionMechanism(attentionArrays: List<DenseNDArray>): AttentionMechanism {

    this.networkProcessor.usedAttentionMechanisms.add(
      AttentionMechanism(attentionArrays, params = this.networkProcessor.model.attentionParams))

    return this.networkProcessor.usedAttentionMechanisms.last()
  }

  /**
   * @param vector the vector that modulates a content-based attention mechanism over the input sequence
   *
   * @return the sequence of attention arrays
   */
  private fun buildAttentionSequence(vector: DenseNDArray): List<DenseNDArray> {

    val transformLayers = this.getTransformLayers(size = this.networkProcessor.inputSequence.size)

    return transformLayers.zip(this.networkProcessor.inputSequence).map { (layer, element) ->

      layer.setInput(0, element)
      layer.setInput(1, vector)
      layer.forward()

      layer.outputArray.values
    }
  }

  /**
   * Get an available group of transform layers, adding it into the usedTransformLayers list.
   *
   * @param size the number of transform layer to build
   *
   * @return an available transform layer
   */
  private fun getTransformLayers(size: Int): List<MergeLayer<DenseNDArray>> {

    this.networkProcessor.usedTransformLayers.add(
      List(size = size, init = { this.networkProcessor.transformLayersPool.getItem() })
    )

    return this.networkProcessor.usedTransformLayers.last()
  }

  /**
   * Initialize the structures used during the forward.
   */
  private fun initForward() {

    this.networkProcessor.transformLayersPool.releaseAll()
    this.networkProcessor.usedTransformLayers.clear()
    this.networkProcessor.usedAttentionMechanisms.clear()
  }
}
