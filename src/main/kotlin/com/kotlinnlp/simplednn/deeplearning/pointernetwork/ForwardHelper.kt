/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.pointernetwork

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.attentionmechanism.AttentionMechanism
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.attentionmechanism.AttentionStructure
import com.kotlinnlp.simplednn.core.mergelayers.affine.AffineLayerStructure
import com.kotlinnlp.simplednn.core.mergelayers.affine.AffineLayersPool
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The forward helper of the [PointerNetwork].
 *
 * @property network the attentive recurrent network of this helper
 */
class ForwardHelper(private val network: PointerNetwork) {

  /**
   * The list of transform layers groups used during the last forward.
   */
  internal val usedTransformLayers = mutableListOf<List<AffineLayerStructure<DenseNDArray>>>()

  /**
   * The list of attention structures used during the last forward.
   */
  internal val usedAttentionStructures = mutableListOf<AttentionStructure>()

  /**
   * A pool of Affine Layers used to build the attention arrays.
   */
  private val transformLayersPool: AffineLayersPool<DenseNDArray> =
    AffineLayersPool(
      inputType = LayerType.Input.Dense,
      activationFunction = Tanh(),
      params = this.network.model.transformParams)

  /**
   * @param decodingInput the input
   *
   * @return an array that contains the importance score for each element of the input sequence
   */
  fun forward(decodingInput: DenseNDArray): DenseNDArray {

    if (this.network.firstState) this.initForward()

    val attentionArrays: List<DenseNDArray> = this.buildAttentionSequence(decodingInput)

    return AttentionMechanism(this.buildAttentionStructure(attentionArrays)).forward()
  }

  /**
   * @param attentionArrays the attention arrays
   *
   * @return the attention structure
   */
  private fun buildAttentionStructure(attentionArrays: List<DenseNDArray>): AttentionStructure {

    this.usedAttentionStructures.add(
      AttentionStructure(attentionArrays, params = this.network.model.attentionParams))

    return this.usedAttentionStructures.last()
  }

  /**
   * @param decodingInput the current decoding input
   *
   * @return the sequence of attention arrays
   */
  private fun buildAttentionSequence(decodingInput: DenseNDArray): List<DenseNDArray> {

    val transformLayers = this.getTransformLayers(size = this.network.inputSequence.size)

    return ArrayList(transformLayers.zip(this.network.inputSequence).map { (layer, encodedElement) ->

      layer.setInput1(encodedElement)
      layer.setInput2(decodingInput)
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

    this.usedTransformLayers.add(
      List(size = size, init = { this.transformLayersPool.getItem() })
    )

    return this.usedTransformLayers.last()
  }

  /**
   * Initialize the structures used during the forward.
   */
  private fun initForward() {

    this.transformLayersPool.releaseAll()
    this.usedTransformLayers.clear()
    this.usedAttentionStructures.clear()
  }
}
