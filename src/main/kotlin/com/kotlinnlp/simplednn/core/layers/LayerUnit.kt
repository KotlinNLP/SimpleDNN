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
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArrayMask
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray

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
   * Assign errors to the parameters associated to this unit. The errors of the output must be already set.
   *
   * gb = errors * 1
   * gw = errors (dot) x
   *
   * @param paramsErrors the parameters associated to this unit
   * @param x the input of the unit
   * @param mePropMask the mask of the top k output nodes, in order to execute the 'meProp' algorithm
   */
  fun assignParamsGradients(paramsErrors: ParametersUnit, x: InputNDArrayType, mePropMask: NDArrayMask? = null) {

    val gw: NDArray<*> = paramsErrors.weights.values
    val gb: NDArray<*> = paramsErrors.biases.values

    if (mePropMask != null) {
      require(x is DenseNDArray) { "Cannot apply 'meProp' method if input is not dense" }
      require(gw is SparseNDArray && gb is SparseNDArray) {
        "Cannot apply 'meProp' method with errors not sparse. Ensure to enable 'meProp' into the params."
      }

      x as DenseNDArray; gw as SparseNDArray; gb as SparseNDArray

      gb.assignValues(this.errors, mask = mePropMask)
      gw.assignDot(this.errors.maskBy(mePropMask), x.T)

    } else {
      gb.assignValues(this.errors)
      gw.assignDot(this.errors, x.T)
    }
  }

  /**
   * Get the errors of the input of the unit. The errors of the output must be already set.
   *
   * @param parameters the parameters associated to this unit
   * @param mePropMask the mask of the top k output nodes, in order to execute the 'meProp' algorithm
   *
   * @return the errors of the input of this unit
   */
  fun getInputErrors(parameters: ParametersUnit, mePropMask: NDArrayMask? = null): DenseNDArray {

    val w: DenseNDArray = parameters.weights.values as DenseNDArray

    return if (mePropMask != null) this.errors.maskBy(mePropMask).T.dot(w) else this.errors.T.dot(w)
  }

  /**
   * Get the relevance of the input of the unit. The relevance of the output must be already set.
   *
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
