/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.randomgenerators

import java.util.*

/**
 * A generator of random numbers uniformly distributed in a closed range centered in 0.0 with a given radius.
 *
 * @property radius radius of the range
 * @property enablePseudoRandom whether to use a pseudo-random generation with the given [seed]
 * @property seed seed used for the pseudo-random generation
 */
class FixedRangeRandom(
  val radius: Double = 0.01,
  val enablePseudoRandom: Boolean = true,
  val seed: Long = 743
) : RandomGenerator {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * A random numbers generator with a uniform distribution.
   */
  private val rndGenerator = if (this.enablePseudoRandom) Random(this.seed) else Random()

  /**
   * @return a random value uniformly distributed in in the range [-[radius], [radius]]
   */
  override fun next(): Double = (2.0 * this.rndGenerator.nextDouble() * this.radius) - this.radius
}
