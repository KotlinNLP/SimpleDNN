/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.pointernetwork

import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.attentionmechanism.AttentionMechanism
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.attentionmechanism.AttentionStructure
import com.kotlinnlp.simplednn.core.mergelayers.affine.AffineLayerStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The forward helper of the [PointerNetwork].
 *
 * @property network the attentive recurrent network of this helper
 */
class ForwardHelper(private val network: PointerNetwork) {

  /**
   * @param input the input
   * @param firstState a boolean indicating if this is the first state
   *
   * @return an array that contains the importance score for each element of the input sequence
   */
  fun forward(input: DenseNDArray,
              firstState: Boolean,
              encodedSequence: List<DenseNDArray>): DenseNDArray {

    val decodingHidden: DenseNDArray = this.network.recurrentProcessor.forward(
      featuresArray = input,
      firstState = firstState)

    val attentionArrays: List<DenseNDArray> = this.buildAttentionSequence(
      encodedSequence = encodedSequence,
      decodingHiddenState = decodingHidden)

    return AttentionMechanism(this.buildAttentionStructure(attentionArrays)).forward()
  }

  /**
   * @param attentionArrays the attention arrays
   *
   * @return the attention structure
   */
  private fun buildAttentionStructure(attentionArrays: List<DenseNDArray>): AttentionStructure {

    this.network.usedAttentionStructures.add(
      AttentionStructure(attentionArrays, params = this.network.model.attentionParams))

    return this.network.usedAttentionStructures.last()
  }

  /**
   * @param encodedSequence the input encoded sequence
   * @param decodingHiddenState the current decoding hidden
   *
   * @return the sequence of attention arrays
   */
  private fun buildAttentionSequence(encodedSequence: List<DenseNDArray>,
                                     decodingHiddenState: DenseNDArray): List<DenseNDArray> {

    val transformLayers = this.getTransformLayers(size = encodedSequence.size)

    return ArrayList(transformLayers.zip(encodedSequence).map { (layer, inputArray) ->

      layer.setInput1(inputArray)
      layer.setInput2(decodingHiddenState)
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
  private fun getTransformLayers(size: Int): List<AffineLayerStructure<DenseNDArray>> {

    this.network.usedTransformLayers.add(
      List(size = size, init = { this.network.transformLayersPool.getItem() })
    )

    return this.network.usedTransformLayers.last()
  }
}
