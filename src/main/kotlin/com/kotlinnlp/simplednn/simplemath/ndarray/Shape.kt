/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.simplemath.ndarray

import java.io.Serializable

/**
 * The shape of an bi-dimensional NDArray containing its dimensions (first and second).
 */
data class Shape(val dim1: Int, val dim2: Int = 1) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The inverse [Shape] of this.
   */
  val inverse: Shape get() = Shape(this.dim2, this.dim1)

  /**
   * @param other any object
   *
   * @return a Boolean indicating if this [Shape] is equal to the given [other] object
   */
  override fun equals(other: Any?): Boolean {
    return (other is Shape && other.dim1 == this.dim1 && other.dim2 == this.dim2)
  }

  /**
   * @return the hash code representation of this [Shape]
   */
  override fun hashCode(): Int {

    var hash = 7
    hash = 83 * hash + this.dim1
    hash = 83 * hash + this.dim2

    return hash
  }
}
