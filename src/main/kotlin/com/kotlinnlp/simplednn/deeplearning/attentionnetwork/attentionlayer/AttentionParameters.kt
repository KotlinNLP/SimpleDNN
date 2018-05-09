/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attentionnetwork.attentionlayer

import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.optimizer.IterableParams
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape

/**
 * Attention parameters.
 *
 * @property attentionSize the size of each array of attention
 * @param initializer the initializer of the context vector (zeros if null, default: Glorot)
 */
class AttentionParameters(
  val attentionSize: Int,
  initializer: Initializer? = GlorotInitializer()
) : IterableParams<AttentionParameters>() {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The context vector trainable parameter.
   */
  val contextVector = UpdatableDenseArray(Shape(this.attentionSize))

  /**
   * The list of all parameters.
   */
  override val paramsList: Array<UpdatableArray<*>> = arrayOf(this.contextVector)

  /**
   * Initialize the values of the context vector.
   */
  init {
    initializer?.initialize(this.contextVector.values)
  }

  /**
   * @return a new [AttentionParameters] containing a copy of all values of this
   */
  override fun copy(): AttentionParameters {

    val clonedParams = AttentionParameters(attentionSize = this.attentionSize, initializer = null)

    clonedParams.contextVector.values.assignValues(this.contextVector.values)

    return clonedParams
  }
}
