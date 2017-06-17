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
    *this.extractLeftContextLabelsFeatures(),
    *this.extractLeftContextFeatures(),
    this.extractFocusFeatures(),
    *this.extractRightContextFeatures()
  )

  /**
   * @return an array containing the features of the elements in the left context
   */
  private fun extractLeftContextFeatures(): Array<DenseNDArray> {
    val leftContext = this.sequence.getLeftContext()

    return Array(size = this.sequence.leftContextSize, init = {
      val i = leftContext[it]
      if (i != null) this.sequence[i] else this.network.emptyVector
    })
  }

  /**
   * @return an array containing the features of the labels of the elements in the left context
   */
  private fun extractLeftContextLabelsFeatures(): Array<DenseNDArray> {
    val leftContext = this.sequence.getLeftContext()

    return Array(size = this.sequence.leftContextSize, init = {
      val i = leftContext[it]
      if (i != null) this.getLabelEmbedding(this.labels[i]).array.values else this.network.emptyLabelVector
    })
  }

  /**
   * @return the features of the focus element
   */
  private fun extractFocusFeatures(): DenseNDArray = this.sequence.getFocusElement()

  /**
   * @return an array containing the features of the elements in the right context
   */
  private fun extractRightContextFeatures(): Array<DenseNDArray> {
    val nextWindow = this.sequence.getRightContext()

    return Array(size = this.sequence.rightContextSize, init = {
      val i = nextWindow[it]
      if (i != null) this.sequence[i] else this.network.emptyVector
    })
  }

  /**
   * @param embeddingIndex an embedding index
   *
   * @return the label embedding representation for a given [embeddingIndex]
   */
  private fun getLabelEmbedding(embeddingIndex: Int) = this.network.labelsEmbeddings.getEmbedding(embeddingIndex)
}
