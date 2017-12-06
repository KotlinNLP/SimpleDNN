/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.initializers

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.activations.ReLU
import com.kotlinnlp.simplednn.core.functionalities.activations.Sigmoid
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.FixedRangeRandom
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import java.util.*

/**
 * An initializer of dense arrays with the 'Glorot Initialization', as explained by Xavier Glorot.
 *
 * References:
 * [Understanding the difficulty of training deep feedforward neural networks]
 * (http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
 *
 * @param gain the gain that determines the scale of the generated values (default = 1.0)
 * @param enablePseudoRandom if true use pseudo-random with a seed (default = true)
 * @param seed seed used for the pseudo-random (default = 743)
 */
class GlorotInitializer(
  private val gain: Double = 1.0,
  private val enablePseudoRandom: Boolean = true,
  private val seed: Long = 743
) : Initializer {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * @param activationFunction the activation function of a layer (can be null)
     *
     * @return the gain to apply to a [GlorotInitializer] in relation to the given [activationFunction]
     */
    fun getGain(activationFunction: ActivationFunction?): Double = when (activationFunction) {
      is ReLU -> 0.5
      is Sigmoid -> 4.0
      else -> 1.0
    }
  }

  /**
   * The random generator of seeds for the pseudo-random initialization.
   */
  private val seedGenerator = Random(this.seed)

  /**
   * Initialize the values of the given [array].
   *
   * @param array a dense array
   */
  override fun initialize(array: DenseNDArray) {

    val randomGenerator = FixedRangeRandom(
      radius = this.gain * Math.sqrt(6.0 / (array.rows + array.columns)),
      enablePseudoRandom = this.enablePseudoRandom,
      seed = this.seedGenerator.nextLong())

    array.randomize(randomGenerator)
  }
}
