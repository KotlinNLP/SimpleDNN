/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.optimizer

import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray
import java.io.Serializable

/**
 * A container of iterable parameters as [UpdatableArray]s, with some utilities methods to assign and copy them.
 */
abstract class IterableParams<SelfType: IterableParams<SelfType>>
  : Serializable, Iterable<com.kotlinnlp.simplednn.core.arrays.UpdatableArray<*>> {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The iterator inner class which iterates over all the parameters of all the layers
   */
  private inner class ParamsIterator: Iterator<UpdatableArray<*>> {

    /**
     * The index of the next parameter.
     */
    private var nextParamIndex: Int = 0

    /**
     * The hasNext() method of the Iterator.
     */
    override fun hasNext(): Boolean {
      return this.nextParamIndex < this@IterableParams.paramsList.lastIndex
    }

    /**
     * The next() method of the Iterator.
     */
    override fun next(): UpdatableArray<*> {
      return this@IterableParams.paramsList[this.nextParamIndex++]
    }
  }

  /**
   * The list of all parameters.
   */
  abstract protected val paramsList: Array<UpdatableArray<*>>

  /**
   * The amount of parameters into this [IterableParams].
   */
  val size: Int get() = this.paramsList.size

  /**
   * @param i the index of a parameter
   *
   * @return the parameter at the given index
   */
  operator fun get(i: Int): UpdatableArray<*> {
    return this.paramsList[i]
  }

  /**
   * The iterator to use to iterate over all the parameters of all the layers
   *
   * @return the iterator of all the parameters
   */
  override fun iterator(): Iterator<UpdatableArray<*>> = this.ParamsIterator()

  /**
   * Assign the values of each parameter of [x] to the parameters of this [IterableParams].
   *
   * @param x the [IterableParams] to assign to this
   */
  fun assignValues(x: SelfType) {

    this.zip(x).forEach {
      it.first.values.assignValues(it.second.values)
    }
  }

  /**
   * Sum the values of each parameter of [x] to the parameters of this [IterableParams].
   *
   * @param x the [IterableParams] to add to this
   */
  fun assignSum(x: SelfType) {

    this.zip(x).forEach {
      val first = it.first.values
      val second = it.second.values

      if (first is SparseNDArray && second is SparseNDArray) {
        first.assignSumMerging(second)
      } else {
        first.assignSum(second)
      }
    }
  }

  /**
   * Divide the values of each parameter by [n].
   *
   * @param n an integer number
   */
  fun assignDiv(n: Int) {

    if (n > 1) {
      val nDouble = n.toDouble()
      this.forEach { it.values.assignDiv(nDouble) }
    }
  }

  /**
   * @return a new [IterableParams] containing a copy of all parameters of this
   */
  abstract fun copy(): SelfType
}
