/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.sequencelabeling

import com.kotlinnlp.simplednn.simplemath.concatVectorsV
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The SWSLFeaturesExtractor is an helper class to extract the relevant features
 * for the neural network [SWSLNetwork] from a given sliding-window sequence.
 *
 * @property sequence the sequence
 * @property labels the labels at the step t-1
 * @property network the neural network
 */
class SWSLFeaturesExtractor(
  private val sequence: SlidingWindowSequence,
  private val labels: ArrayList<Int>,
  private val network: SWSLNetwork) {

  /**
   * Return the features extracted in the current state of the sliding-window sequence
   *
   * The features are obtained by a vector concatenation of
   *    - the dense representation of the previous labels embeddings
   *    - the dense representation of the prev elements
   *    - the dense representation of the focus element
   *    - the dense representation of the next elements
   *
   *  @return the features of the current state
   */
  fun getFeatures(): DenseNDArray = concatVectorsV(
    *this.extractPrevLabelsFeatures(),
    *this.extractPrevContextFeatures(),
    this.extractFocusFeatures(),
    *this.extractNextContextFeatures()
  )

  /**
   * @return the dense-representation of the focus element
   */
  private fun extractFocusFeatures(): DenseNDArray = this.sequence.getFocus()

  /**
   * @return the dense-representations of the next elements
   */
  private fun extractNextContextFeatures(): Array<DenseNDArray> {
    val nextWindow = this.sequence.getRightContext()

    return Array(size = this.sequence.rightContextSize, init = {
      val i = nextWindow[it]
      if (i != null) this.sequence[i] else this.network.emptyVector
    })
  }

  /**
   * @return the dense-representations of the previous elements
   */
  private fun extractPrevContextFeatures(): Array<DenseNDArray> {
    val prevWindow = this.sequence.getLeftContext()

    return Array(size = this.sequence.leftContextSize, init = {
      val i = prevWindow[it]
      if (i != null) this.sequence[i] else this.network.emptyVector
    })
  }

  /**
   * @return the dense-representations of the previous labels
   */
  private fun extractPrevLabelsFeatures(): Array<DenseNDArray> {
    val prevWindow = this.sequence.getLeftContext()

    return Array(size = this.sequence.leftContextSize, init = {
      val i = prevWindow[it]
      if (i != null) this.getLabelEmbedding(this.labels[i]).array.values else this.network.emptyLabelVector
    })
  }

  /**
   * Return the label embedding representation for a given [embeddingIndex]
   *
   * @param embeddingIndex an embedding index
   *
   * @return the label embedding for the index [embeddingIndex]
   */
  private fun getLabelEmbedding(embeddingIndex: Int) = this.network.labelsEmbeddings.getEmbedding(embeddingIndex)
}
