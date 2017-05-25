/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.simplemath.ndarray

/**
 *
 */
class NDArrayMask(val dim1: Array<Int>, val dim2: Array<Int>) : Iterable<Pair<Int, Int>> {

  init {
    require(dim1.size == dim2.size)
  }

  /**
   *
   */
  private inner class NDArrayMaskIterator : Iterator<Pair<Int, Int>> {

    /**
     *
     */
    private var curIndex = 0

    /**
     *
     */
    override fun hasNext(): Boolean = curIndex < this@NDArrayMask.dim1.size

    /**
     *
     */
    override fun next(): Pair<Int, Int> {

      val next = Pair(this@NDArrayMask.dim1[curIndex], this@NDArrayMask.dim2[curIndex])

      this.curIndex++

      return next
    }

  }

  /**
   *
   */
  override fun iterator(): Iterator<Pair<Int, Int>> = this.NDArrayMaskIterator()
}
