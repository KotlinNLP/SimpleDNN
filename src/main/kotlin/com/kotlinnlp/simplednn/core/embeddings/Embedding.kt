/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.embeddings

import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
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
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * Build a new [Embedding] with the given [id] and [vector].
   *
   * @param id the id of the embedding
   * @param vector the vector of the embedding
   *
   * @return a new embedding
   */
  constructor(id: Int, vector: DoubleArray): this(
    id = id,
    array = UpdatableDenseArray(values = DenseNDArrayFactory.arrayOf(vector))
  )

  /**
   * Build a new [Embedding] with the given [id] and [vector].
   *
   * @param id the id of the embedding
   * @param vector the vector of the embedding
   *
   * @return a new embedding
   */
  constructor(id: Int, vector: DenseNDArray): this(
    id = id,
    array = UpdatableDenseArray(values = vector)
  )

  /**
   * Build a new [Embedding] with the given [id] and [size].
   *
   * @param id the id of the embedding
   * @param size the size of the embedding
   * @param initializer the initializer of the values (can be null)
   *
   * @return a new [Embedding] with the given [id]
   */
  constructor(id: Int, size: Int, initializer: Initializer?): this(
    id = id,
    array = UpdatableDenseArray(Shape(size)).apply { initializer?.initialize(values) }
  )

  /**
   * @param digits precision specifier
   *
   * @return a string representation of the [array], concatenating the elements with the space character.
   */
  fun toString(digits: Int): String {

    val sb = StringBuilder()

    (0 until this.array.values.length).forEach {
      sb.append(" ").append("%.${digits}f".format(this.array.values[it]))
    }

    return sb.toString()
  }
}
