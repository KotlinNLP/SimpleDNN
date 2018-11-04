/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers

import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.core.arrays.UpdatableSparseArray
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.optimizer.IterableParams
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The parameters of a layer
 *
 * @property inputSize input size
 * @property outputSize output size
 * @param weightsInitializer the initializer of the weights (zeros if null)
 * @param biasesInitializer the initializer of the biases (zeros if null)
 */
abstract class LayerParameters<SelfType: LayerParameters<SelfType>>(
  val inputSize: Int,
  val outputSize: Int,
  private val weightsInitializer: Initializer?,
  private val biasesInitializer: Initializer?
) : IterableParams<SelfType>() {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The list of weights parameters.
   */
  protected abstract val weightsList: List<UpdatableArray<*>>

  /**
   * The list of biases parameters.
   */
  protected abstract val biasesList: List<UpdatableArray<*>>

  /**
   * Initialize the values of the parameters with the given [weightsInitializer] and [biasesInitializer].
   * If an initializer is null, its related parameters are initialized to zeros.
   *
   * Note: this method should be called into the 'init' block.
   */
  protected fun initialize() {

    if (this.weightsInitializer != null) {
      require(this.weightsList.all { it is UpdatableDenseArray }) { "Cannot initialize weights not dense" }
      this.weightsList.forEach { this.weightsInitializer.initialize(it.values as DenseNDArray) }
    }

    if (this.biasesInitializer != null) {
      require(this.biasesList.all { it is UpdatableDenseArray }) { "Cannot initialize biases not dense" }
      this.biasesList.forEach { this.biasesInitializer.initialize(it.values as DenseNDArray) }
    }
  }

  /**
   *
   */
  protected fun buildUpdatableArray(dim1: Int, dim2: Int = 1, sparseInput: Boolean = false): UpdatableArray<*> =
    if (sparseInput)
      this.buildSparseArray(dim1, dim2)
    else
      this.buildDenseArray(dim1, dim2)

  /**
   *
   */
  protected fun buildDenseArray(dim1: Int, dim2: Int = 1) = UpdatableDenseArray(Shape(dim1, dim2))

  /**
   *
   */
  private fun buildSparseArray(dim1: Int, dim2: Int = 1) = UpdatableSparseArray(Shape(dim1, dim2))
}
