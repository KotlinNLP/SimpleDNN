/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.dataset

import java.util.*

/**
 *
 */
class Shuffler(enablePseudoRandom: Boolean = true, seed: Long = 743) {
  private val rndGenerator = if (enablePseudoRandom) Random(seed) else Random()

  /**
   *
   */
  private fun swap(xs: IntArray, i: Int, j: Int) {
    val t = xs[i]
    xs[i] = xs[j]
    xs[j] = t
  }

  /**
   *
   */
  operator fun invoke(xs: IntArray) {
    for (i in xs.indices.reversed()) {
      swap(xs, i, rndGenerator.nextInt(i + 1))
    }
  }
}
