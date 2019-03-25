/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.helpers.RelevanceUtils
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * Forward from the given input.
 *
 * g = w (dot) x + b
 *
 * @param w the weights (must be Dense)
 * @param b the biases
 * @param x the input array of the current layer
 */
fun <InputNDArrayType : NDArray<InputNDArrayType>>AugmentedArray<*>.forward(
  w: NDArray<*>, b: NDArray<*>, x: InputNDArrayType
) = this.values.assignDot(w as DenseNDArray, x).assignSum(b)

/**
 * Assign errors to the parameters associated to this unit. The errors of the output must be already set.
 *
 * gb = errors * 1
 * gw = errors (dot) x
 *
 * @param gw the gradients of the weights
 * @param gb the gradients of the biases
 * @param x the input of the unit
 */
fun <InputNDArrayType : NDArray<InputNDArrayType>>AugmentedArray<*>.assignParamsGradients(
  gw: NDArray<*>, gb: NDArray<*>, x: InputNDArrayType
) {
  gb.assignValues(this.errors)
  gw.assignDot(this.errors, x.t)
}

/**
 * Get the errors of the input of the unit. The errors of the output must be already set.
 *
 * @param w the weights
 *
 * @return the errors of the input of this unit
 */
fun AugmentedArray<DenseNDArray>.getInputErrors(w: NDArray<*>): DenseNDArray = this.errors.t.dot(w)

/**
 * Get the relevance of the input of the unit. The relevance of the output must be already set.
 *
 * @param x the input of the unit
 * @param cw the weights-contribution of the input to calculate the output
 *
 * @return the relevance of the input of the unit
 */
fun AugmentedArray<DenseNDArray>.getInputRelevance(x: DenseNDArray, cw: DenseNDArray): DenseNDArray =
  RelevanceUtils.calculateRelevanceOfArray(
  x = x,
  y = this.valuesNotActivated,
  yRelevance = this.relevance,
  contributions = cw
)
