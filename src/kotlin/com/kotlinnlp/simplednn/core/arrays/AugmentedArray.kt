/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.arrays

import com.kotlinnlp.simplednn.simplemath.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape

/**
 * The AugmentedArray extends an ActivableArray with [errors] properties.
 *
 * @property size the length of the array
 */
open class AugmentedArray(size: Int) : ActivableArray(size) {

  /**
   *
   */
  companion object {

    /**
     *
     * @param values the initial values to assign to the AugmentedArray
     * @return an AugmentedArray with the array already initialized
     */
    operator fun invoke(values: NDArray): AugmentedArray {

      val array = AugmentedArray(size = values.length)

      array.assignValues(values)

      return array
    }
  }

  /**
   * Contains the errors on the current values
   */
  val errors: NDArray = NDArray.zeros(Shape(size))

  /**
   * Assign errors to the array
   *
   * @param errors errors to assign to this AugmentedArray.
   *               The errors must have the same size of the array values.
   */
  fun assignErrors(errors: NDArray) { this.errors.assignValues(errors) }

  /**
   *
   * @return a clone of this AugmentedArray
   */
  override fun clone(): AugmentedArray {

    val clonedArray = AugmentedArray(this.size)

    clonedArray._values.assignValues(this._values)

    if (this.hasActivation) {
      clonedArray._valuesNotActivated = this.valuesNotActivated.copy()
      clonedArray.setActivation(this.activationFunction!!)
    }

    clonedArray.errors.assignValues(this.errors)

    return clonedArray
  }
}
