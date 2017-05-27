/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.arrays

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.simplemath.ndarray.*

/**
 * The [ActivableArray] is a wrapper of an [NDArray] in which values are modified according
 * to the activation function being used (e.g. Tanh, Sigmoid, ReLU, ELU).
 *
 * @property size the size of the [values]
 */
open class ActivableArray<NDArrayType : NDArray<NDArrayType>>(val size: Int) {

  companion object {

    /**
     * [ActivableArray] factory by values.
     *
     * @param values the initial values to assign to the [ActivableArray]
     *
     * @return an [ActivableArray] with the values already initialized
     */
    operator fun <T : NDArray<T>> invoke(values: T): ActivableArray<T> {

      val array = ActivableArray<T>(size = values.length)

      array.assignValues(values)

      return array
    }
  }

  /**
   * An [NDArray] containing the values of this [ActivableArray]
   */
  val values get() = this._values

  /**
   * An [NDArray] containing the values of this [ActivableArray]
   */
  lateinit protected var _values: NDArrayType

  /**
   * An [NDArray] containing the values not activated of this [ActivableArray] (respect on the last call of activate())
   */
  val valuesNotActivated: NDArrayType get() =
    if (this._valuesNotActivated != null)
      this._valuesNotActivated!!
    else
      this._values

  /**
   * The function used to activate this [ActivableArray] (e.g. Tanh, Sigmoid, ReLU, ELU)
   */
  protected var activationFunction: ActivationFunction? = null

  /**
   * Whether this array has an activation function
   */
  val hasActivation: Boolean get() = this.activationFunction != null

  /**
   * An [NDArray] containing the values not activated of this [ActivableArray] (respect on the last call of activate())
   */
  protected var _valuesNotActivated: NDArrayType? = null

  /**
   * Assign values to the array
   * @param values values to assign to this [ActivableArray]
   */
  fun assignValues(values: NDArrayType) {
    try {
      this._values.assignValues(values)
    } catch (e: UninitializedPropertyAccessException) {
      this._values = values.copy()
    }
  }

  /**
   *
   * @return set the activation function of this [ActivableArray]
   */
  open fun setActivation(activationFunction: ActivationFunction) {
    this.activationFunction = activationFunction
  }

  /**
   * Activate the array memorizing the values not activated and setting the new activated values
   */
  fun activate() {

    if (this.hasActivation) {

      if (this._valuesNotActivated == null) {
        this._valuesNotActivated = this._values.copy()
      } else {
        this._valuesNotActivated!!.assignValues(this._values)
      }

      this._values.assignValues(this.activationFunction!!.f(this._valuesNotActivated as DenseNDArray))
    }
  }

  /**
   * Activate the array without modifying it, but only returning the values
   * @return the activated values
   */
  fun getActivatedValues(): DenseNDArray {
    return this.activationFunction!!.f(this.valuesNotActivated as DenseNDArray)
  }

  /**
   *
   * @return the derivative of the activation calculated in valuesNotActivated (it uses an
   *         optimized function because all the common functions used as activation contain
   *         the activated values themselves in their derivative)
   */
  fun calculateActivationDeriv(): DenseNDArray {
    return this.activationFunction!!.dfOptimized(this._values as DenseNDArray)
  }

  /**
   *
   * @return a clone of this [ActivableArray]
   */
  open fun clone(): ActivableArray<NDArrayType> {

    val clonedArray = ActivableArray<NDArrayType>(size = this._values.length)
    clonedArray.assignValues(this.values)

    if (this.hasActivation) {

      if (this._valuesNotActivated != this._values) {
        clonedArray._valuesNotActivated = this._valuesNotActivated!!.copy()
      }

      clonedArray.setActivation(this.activationFunction!!)
    }

    return clonedArray
  }
}
