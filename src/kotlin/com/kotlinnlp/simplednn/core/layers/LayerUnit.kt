/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape

/**
 * The basic unit of the layer, which extends the [AugmentedArray] with forward and backward methods.
 */
open class LayerUnit<InputNDArrayType : NDArray<InputNDArrayType>>(size: Int) : AugmentedArray<DenseNDArray>(size) {

  /**
   * Initialize values with an empty array.
   */
  init {
    this.assignValues(DenseNDArrayFactory.emptyArray(Shape(size)))
  }

  /**
   * Forward from the given input.
   *
   * g = w (dot) x + b
   *
   * @param parameters the parameters associated to this unit
   * @param x the input array of the current layer
   */
  fun forward(parameters: ParametersUnit, x: InputNDArrayType) {

    val w = parameters.weights.values as DenseNDArray
    val b = parameters.biases.values

    this.values.assignDot(w, x).assignSum(b)
  }

  /**
   * Assign errors to the parameters associated to this unit.
   *
   * gb = errors * 1
   * gw = errors (dot) x
   *
   * @param paramsErrors the parameters associated to this unit
   * @param x the input of the unit
   */
  fun assignParamsGradients(paramsErrors: ParametersUnit, x: InputNDArrayType) {

    val gb: DenseNDArray = paramsErrors.biases.values
    val gw: NDArray<*> = paramsErrors.weights.values

    gb.assignValues(this.errors)
    gw.assignDot(this.errors, x.T)
  }

  /**
   * @param parameters the parameters associated to this unit
   *
   * @return the errors of the input of this unit
   */
  fun getInputErrors(parameters: ParametersUnit): DenseNDArray {

    val w: DenseNDArray = parameters.weights.values as DenseNDArray

    return this.errors.T.dot(w)
  }

  /**
   * @param x the input of the unit
   * @param contributions the contribution of the input to calculate the output
   *
   * @return the relevance of the input of the unit
   */
  fun getInputRelevance(x: InputNDArrayType, contributions: ParametersUnit): NDArray<*> {

    return RelevanceUtils.calculateRelevanceOfArray(
      x = x,
      y = this.valuesNotActivated,
      yRelevance = this.relevance as DenseNDArray,
      contributions = contributions.weights.values
    )
  }
}
