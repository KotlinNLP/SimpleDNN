/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.embeddings

import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.simplemath.format
import java.io.Serializable

/**
 * An Embedding is a dense vectors of real numbers.
 *
 * @property id the id of the Embedding in the lookupTable
 * @property array the values of the Embedding
 */
data class Embedding(val id: Int, val array: UpdatableDenseArray) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * @param digits precision specifier
   *
   * @return a string representation of the [array], concatenating the elements with the space character.
   */
  fun toString(digits: Int): String {

    val sb = StringBuilder()

    (0 until this.array.values.length).forEach {
      sb.append(" ").append(this.array.values[it].format(digits))
    }

    return sb.toString()
  }
}
