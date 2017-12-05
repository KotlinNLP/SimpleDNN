/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.initializers

import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.FixedRangeRandom
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * An initializer of dense arrays with the 'Glorot Initialization', as explained by Xavier Glorot.
 *
 * References:
 * [Understanding the difficulty of training deep feedforward neural networks]
 * (http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
 *
 * @param enablePseudoRandom if true use pseudo-random with a seed
 * @param seed seed used for the pseudo-random
 */
class GlorotInitializer(private val enablePseudoRandom: Boolean = true, private val seed: Long = 743) : Initializer {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * Initialize the values of the given [array].
   *
   * @param array a dense array
   */
  override fun initialize(array: DenseNDArray) {

    val randomGenerator = FixedRangeRandom(
      radius = Math.sqrt(6.0 / (array.rows + array.columns)),
      enablePseudoRandom = this.enablePseudoRandom,
      seed = this.seed)

    array.randomize(randomGenerator)
  }
}
