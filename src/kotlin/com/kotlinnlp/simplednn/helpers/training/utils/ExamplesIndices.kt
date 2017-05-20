/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.helpers.training.utils

import com.kotlinnlp.simplednn.dataset.Shuffler

/**
 *
 */
class ExamplesIndices(val size: Int, val shuffler: Shuffler?): Iterable<Int> {

  private inner class IndicesIterator : Iterator<Int> {
    /**
     *
     */
    private var count: Int = 0

    /**
     *
     */
    override fun hasNext(): Boolean = this.count < (ExampleIndices@size)

    /**
     *
     */
    override fun next(): Int = ExampleIndices@indices[this.count++]
  }

  /**
   *
   */
  val indices = IntArray(size = size, init = { it })

  init { shuffler?.invoke(indices) }

  /**
   *
   */
  override fun iterator(): Iterator<Int> = IndicesIterator()
}
