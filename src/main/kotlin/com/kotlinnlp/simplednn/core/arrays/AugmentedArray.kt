/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.arrays

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

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
  private lateinit var _errors: DenseNDArray

  /**
   * Contains the relevance of the current values
   */
  val relevance: NDArray<*> get() = this._relevance

  /**
   * Contains the relevance of the current values
   */
  private lateinit var _relevance: NDArray<*>

  /**
   * Contains the relevance of the current values when involved in recurrent calculations
   */
  val recurrentRelevance: NDArray<*> get() = this._recurrentRelevance

  /**
   * Contains the relevance of the current values when involved in recurrent calculations
   */
  private lateinit var _recurrentRelevance: NDArray<*>

  /**
   * Assign errors to the array.
   *
   * @param errors the [DenseNDArray] to assign as errors to this [AugmentedArray]. It must have the same size
   *               of the values of this [AugmentedArray].
   *
   * @return the errors of this [AugmentedArray]
   */
  fun assignErrors(errors: DenseNDArray): DenseNDArray {

    require(errors.length == this.size) { "Errors must have the same size of the values" }

    if (::_errors.isInitialized) {

      this._errors.assignValues(errors)

    } else {

      this._errors = if (this._values.columns == 1 && errors.rows == 1 || this._values.rows == 1 && errors.columns == 1)
        errors.t // Assignment between row and column vectors
      else
        errors.copy()
    }

    return this._errors
  }

  /**
   * Assign errors as array of zeros.
   * This method optimizes the calculation avoiding the creation of a new [NDArray] when [errors] are already set.
   * If [errors] are still not initialized a new [NDArray] is created.
   *
   * @return the errors of this [AugmentedArray]
   */
  fun assignZeroErrors(): DenseNDArray {

    if (::_errors.isInitialized)
      this._errors.zeros()
    else
      this._errors = DenseNDArrayFactory.zeros(Shape(this.size))

    return this._errors
  }

  /**
   * Assign errors as product of [a] and [b].
   * This method optimizes the calculation avoiding the creation of a new [NDArray] when [errors] are already set.
   * If [errors] are still not initialized a new [NDArray] is created.
   *
   * @param a the first [DenseNDArray] factor
   * @param b the second [DenseNDArray] factor
   *
   * @return the errors of this [AugmentedArray]
   */
  fun assignErrorsByProd(a: DenseNDArray, b: DenseNDArray): DenseNDArray {

    require(a.length == this.size) { "Invalid arrays size" }

    if (::_errors.isInitialized)
      this._errors.assignProd(a, b)
    else
      this._errors = a.prod(b)

    return this._errors
  }

  /**
   * Assign errors as product of [a] and [b].
   * This method optimizes the calculation avoiding the creation of a new [NDArray] when [errors] are already set.
   * If [errors] are still not initialized a new [NDArray] is created.
   *
   * @param a the first factor, as [DenseNDArray]
   * @param b the second factor, as Double number
   *
   * @return the errors of this [AugmentedArray]
   */
  fun assignErrorsByProd(a: DenseNDArray, b: Double): DenseNDArray {

    require(a.length == this.size) { "Invalid arrays size" }

    if (::_errors.isInitialized)
      this._errors.assignProd(a, b)
    else
      this._errors = a.prod(b)

    return this._errors
  }

  /**
   * Assign errors as dot product of [a] by [b].
   * This method optimizes the calculation avoiding the creation of a new [NDArray] when [errors] are already set.
   * If [errors] are still not initialized a new [NDArray] is created.
   *
   * @param a the first [DenseNDArray] factor
   * @param b the second [DenseNDArray] factor
   *
   * @return the errors of this [AugmentedArray]
   */
  fun assignErrorsByDot(a: DenseNDArray, b: NDArray<*>): DenseNDArray {

    if (::_errors.isInitialized)
      this._errors.assignDot(a, b)
    else
      this._errors = a.dot(b)

    return this._errors
  }

  /**
   * Assign errors as the transpose of the dot product of [a] by [b].
   * This method optimizes the calculation avoiding the creation of a new [NDArray] when [errors] are already set.
   * If [errors] are still not initialized a new [NDArray] is created.
   *
   * @param a the first [DenseNDArray] factor
   * @param b the second [DenseNDArray] factor
   *
   * @return the errors of this [AugmentedArray]
   */
  fun assignErrorsByDotT(a: DenseNDArray, b: NDArray<*>): DenseNDArray {

    if (::_errors.isInitialized)
      this._errors.assignValues(a.dot(b))
    else
      this._errors = a.dot(b).t

    return this._errors
  }

  /**
   * Assign the relevance to the array.
   *
   * @param relevance the [NDArray] to assign to this [AugmentedArray] as relevance.
   *                  It must have the same size of the values of this [AugmentedArray].
   */
  fun assignRelevance(relevance: NDArray<*>) {

    require(relevance.length == this.size) { "Relevance must have the same size of the values" }

    if (::_relevance.isInitialized)
      this._relevance.assignValues(relevance)
    else
      this._relevance = relevance.copy()
  }

  /**
   * Assign the recurrent relevance to the array.
   *
   * @param relevance the [NDArray] to assign to this [AugmentedArray] as recurrent relevance.
   *                  It must have the same size of the values of this [AugmentedArray].
   */
  fun assignRecurrentRelevance(relevance: NDArray<*>) {

    require(relevance.length == this.size) { "Relevance must have the same size of the values" }

    if (::_recurrentRelevance.isInitialized)
      this._recurrentRelevance.assignValues(relevance)
    else
      this._recurrentRelevance = relevance.copy()
  }

  /**
   * Assign values to the array.
   * WARNING: this operation resets the errors too if they were assigned!
   *
   * @param values values to assign to this [ActivableArray]
   */
  override fun assignValues(values: NDArrayType) {

    super.assignValues(values)

    if (::_errors.isInitialized) this._errors.zeros()
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

    if (::_errors.isInitialized) clonedArray.assignErrors(this._errors)
    if (::_relevance.isInitialized) clonedArray.assignRelevance(this._relevance)
    if (::_recurrentRelevance.isInitialized) clonedArray.assignRecurrentRelevance(this._recurrentRelevance)

    return clonedArray
  }
}
