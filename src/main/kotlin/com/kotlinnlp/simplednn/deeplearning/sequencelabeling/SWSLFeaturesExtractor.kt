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
 * The SWSLFeaturesExtractor is a helper to extract the relevant features for the neural network [SWSLNetwork] from a
 * given sliding-window [sequence].
 *
 * @property sequence the sequence
 * @property labels the labels at the step t-1
 * @property network the neural network
 */
class SWSLFeaturesExtractor(
  private val sequence: SlidingWindowSequence,
  private val labels: List<SWSLabeler.Label>,
  private val network: SWSLNetwork) {

  /**
   * Get the features extracted from current state of the sliding-window sequence.
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
   * @return an array containing the features of the labels of the elements in the left context
   */
  private fun extractLeftContextLabelsFeatures(): Array<DenseNDArray> {

    val leftContext = this.sequence.getLeftContext()

    return Array(
      size = this.sequence.leftContextSize,
      init = {
        val i = leftContext[it]
        if (i != null) this.getLabelEmbedding(this.labels[i].index).array.values else this.network.emptyLabelVector
      }
    )
  }

  /**
   * @return an array containing the features of the elements in the left context
   */
  private fun extractLeftContextFeatures(): Array<DenseNDArray> {

    val leftContext = this.sequence.getLeftContext()

    return Array(
      size = this.sequence.leftContextSize,
      init = { i -> this.getWindowElement(index = leftContext[i]) }
    )
  }

  /**
   * @return the features of the focus element
   */
  private fun extractFocusFeatures(): DenseNDArray = this.sequence.getFocusElement()

  /**
   * @return an array containing the features of the elements in the right context
   */
  private fun extractRightContextFeatures(): Array<DenseNDArray> {

    val rightContext = this.sequence.getRightContext()

    return Array(
      size = this.sequence.rightContextSize,
      init = { i -> this.getWindowElement(index = rightContext[i]) }
    )
  }

  /**
   * @param labelIndex a label index
   *
   * @return the label embedding representation for a given [labelIndex]
   */
  private fun getLabelEmbedding(labelIndex: Int) = this.network.labelsEmbeddings.getOrSet(labelIndex)

  /**
   * Get an element of the features window.
   *
   * @param index the index of an element (can be null)
   *
   * @return the element of the sequence at the given [index] or the emptyVector if [index] is null
   */
  private fun getWindowElement(index: Int?): DenseNDArray {
    return if (index != null) this.sequence[index] else this.network.emptyVector
  }
}
