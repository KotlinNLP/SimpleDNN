/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.randomgenerators

import java.util.*

/**
 * A generator of random numbers uniformly distributed in the range [0.0, 1.0].
 *
 * @property enablePseudoRandom whether to use a pseudo-random generation with the given [seed]
 * @property seed seed used for the pseudo-random generation
 */
class BaseRandom(val enablePseudoRandom: Boolean = true, val seed: Long = 743) : RandomGenerator {

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
   * @return a random value uniformly distributed in in the range [0.0, 1.0]
   */
  override fun next(): Double = this.rndGenerator.nextDouble()
}
