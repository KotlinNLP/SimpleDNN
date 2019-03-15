/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.arrays

import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArrayFactory
import java.util.UUID

/**
 * The [ParamsArray] is a wrapper of an [UpdatableArray] extending it with an unique identifier [uuid] and methods to
 * build the params [Errors].
 *
 * @property values the values of the parameters
 */
class ParamsArray(values: DenseNDArray) : UpdatableDenseArray(values) {

  companion object {

    /**
     * Return a new [ParamsArray] with the same values and updaterSupportStructure of the given [array].
     *
     * @param array an updatable array
     *
     * @return a new params array
     */
    operator fun invoke(array: UpdatableDenseArray) = ParamsArray(array.values).apply {
      updaterSupportStructure = array.updaterSupportStructure
    }
  }

  /**
   * Build a new [ParamsArray] with the given [values].
   *
   * @param values the values
   *
   * @return a new params array
   */
  constructor(values: DoubleArray) : this(DenseNDArrayFactory.arrayOf(values))

  /**
   * Build a new [ParamsArray] with the given [values].
   *
   * @param values the values
   *
   * @return a new params array
   */
  constructor(values: List<DoubleArray>) : this(DenseNDArrayFactory.arrayOf(values))

  /**
   * Build a new [ParamsArray] with the given [shape].
   *
   * @param shape the shape
   * @param initializer the initializer of the values (can be null)
   *
   * @return a new params array
   */
  constructor(shape: Shape, initializer: Initializer?) : this(
    values = DenseNDArrayFactory.zeros(shape).apply { initializer?.initialize(this) }
  )

  /**
   * Build a new [ParamsArray] with the given [size].
   *
   * @param size the size
   * @param initializer the initializer of the values (can be null)
   *
   * @return a new params array
   */
  constructor(size: Int, initializer: Initializer?) : this(
    values = DenseNDArrayFactory.zeros(Shape(size)).apply { initializer?.initialize(this) }
  )

  /**
   * The unique identifier of this [ParamsArray].
   */
  val uuid = UUID.randomUUID().toString()

  /**
   * ParamsErrors.
   *
   * @property values the error of the parameters
   */
  inner class Errors<T: NDArray<T>>(val values: T) {

    /**
     * Reference parameters.
     *
     * The instance of the [ParamsArray] from witch the [Errors] has been created.
     */
    val refParams: ParamsArray = this@ParamsArray

    /**
     * @return a copy of this params errors (the copy share the same [refParams])
     */
    fun copy() = Errors(this.values.copy())
  }

  /**
   * Return a new instance of [Errors] initialized to zeros or with the given [values] if not null.
   *
   * @param values the values used to initialize the errors (can be null)
   *
   * @return a new instance of errors parameters
   */
  fun buildDenseErrors(values: DenseNDArray? = null) =
    Errors(values ?: DenseNDArrayFactory.zeros(this.values.shape))

  /**
   * Return a new instance of [Errors] initialized to zeros or with the given [values] if not null.
   *
   * @param values the values used to initialize the errors (can be null)
   *
   * @return a new instance of errors parameters
   */
  fun buildSparseErrors(values: SparseNDArray? = null) =
    Errors(values ?: SparseNDArrayFactory.zeros(this.values.shape))
}
