/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.helpers.training.utils

import com.kotlinnlp.simplednn.dataset.Shuffler

/**
 * An helper class to iterate over integer indices shuffling them with a [Shuffler] object.
 *
 * @property size the number of indices over which to iterate
 * @param shuffler the optional [Shuffler] object (if null indices are not shuffled)
 */
class ExamplesIndices(private val size: Int, shuffler: Shuffler? = null) : Iterable<Int> {

  /**
   * Indices Iterator class.
   */
  private inner class IndicesIterator : Iterator<Int> {

    /**
     * The count of the current index.
     */
    private var count: Int = 0

    /**
     * Iterator hasNext method.
     */
    override fun hasNext(): Boolean = this.count < (ExampleIndices@size)

    /**
     * Iterator next method.
     */
    override fun next(): Int = ExampleIndices@indices[this.count++]
  }

  /**
   * The
   */
  private val indices = IntArray(size = size, init = { it })

  /**
   * Indices shuffle.
   */
  init {
    shuffler?.invoke(indices)
  }

  /**
   * @return the iterator over the indices
   */
  override fun iterator(): Iterator<Int> = IndicesIterator()
}
