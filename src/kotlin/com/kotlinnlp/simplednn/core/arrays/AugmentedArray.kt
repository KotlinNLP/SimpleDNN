/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.arrays

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * The [AugmentedArray] extends the [ActivableArray] with the errors and relevance.
 *
 * @property size the length of the array
 */
open class AugmentedArray<NDArrayType : NDArray<NDArrayType>>(size: Int) : ActivableArray<NDArrayType>(size) {

  /**
   *
   */
  companion object {

    /**
     * [AugmentedArray] factory by values.
     *
     * @param values the initial values to assign to the [AugmentedArray]
     *
     * @return an [AugmentedArray] with the values already initialized
     */
    operator fun <T : NDArray<T>> invoke(values: T): AugmentedArray<T> {

      val array = AugmentedArray<T>(size = values.length)

      array.assignValues(values)

      return array
    }
  }

  /**
   * Contains the errors of the current values
   */
  val errors: DenseNDArray get() = this._errors

  /**
   * Contains the errors of the current values
   */
  lateinit protected var _errors: DenseNDArray

  /**
   * Contains the relevance of the current values
   */
  val relevance: Norm1Array<*> get() = this._relevance

  /**
   * Contains the relevance of the current values
   */
  lateinit protected var _relevance: Norm1Array<*>

  /**
   * Assign errors to the array.
   *
   * @param errors the [DenseNDArray] to assign as errors to this [AugmentedArray]. It must have the same size
   *               of the values of this [AugmentedArray].
   */
  fun assignErrors(errors: DenseNDArray) {
    require(errors.length == this.size) { "Errors must have the same size of the values" }

    try {
      this._errors.assignValues(errors)

    } catch (e: UninitializedPropertyAccessException) {
      require(errors.length == this.size)

      this._errors = errors.copy()
    }
  }

  /**
   * Assign the relevance to the array.
   *
   * @param relevance the [Norm1Array] to assign to this [AugmentedArray]. It must have the same size
   *                  of the values of this [AugmentedArray].
   */
  fun assignRelevance(relevance: Norm1Array<*>) {
    require(relevance.length == this.size) { "Relevance must have the same size of the values" }

    try {
      this._relevance.assignValues(relevance.values)

    } catch (e: UninitializedPropertyAccessException) {
      require(relevance.length == this.size)

      this._relevance = relevance.clone()
    }
  }

  /**
   * Assign values to the array. WARNING: this operation resets the errors too!
   *
   * @param values values to assign to this [ActivableArray]
   */
  override fun assignValues(values: NDArrayType) {
    super.assignValues(values)
    this._errors = DenseNDArrayFactory.emptyArray(values.shape)
  }

  /**
   * Clone this [AugmentedArray].
   *
   * @return a clone of this [AugmentedArray]
   */
  override fun clone(): AugmentedArray<NDArrayType> {

    val clonedArray = AugmentedArray<NDArrayType>(this.size)

    try {
      clonedArray.assignValues(this._values)
    } catch (e: UninitializedPropertyAccessException) {}

    if (this.hasActivation) {

      if (this._valuesNotActivated != null && this._valuesNotActivated != this._values) {
        clonedArray._valuesNotActivated = this._valuesNotActivated!!.copy()
      }

      clonedArray.setActivation(this.activationFunction!!)
    }

    try {
      clonedArray.assignErrors(this._errors)
    } catch (e: UninitializedPropertyAccessException) {}

    try {
      clonedArray.assignRelevance(this._relevance)
    } catch (e: UninitializedPropertyAccessException) {}

    return clonedArray
  }
}
