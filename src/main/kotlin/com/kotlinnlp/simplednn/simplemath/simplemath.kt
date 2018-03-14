/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.simplemath

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape

/**
 * Equals within tolerance.
 *
 * @param a a [Double] number
 * @param b a [Double] number
 * @param tolerance it defines the range [[b] - [tolerance], [b] + [tolerance]]
 *
 * @return a [Boolean] which indicates if [a] is equal to [b] within the [tolerance]
 */
fun equals(a: Double, b: Double, tolerance: Double = 10e-4): Boolean {

  val lower = b - tolerance
  val upper = b + tolerance

  return a in lower..upper
}

/**
 * Equals within tolerance.
 *
 * @param a an array of [Double] numbers
 * @param b an array of [Double] numbers
 * @param tolerance it defines the range [elm - [tolerance], elm + [tolerance]]
 *
 * @return a [Boolean] which indicates if all the elements of [a] are equal to the
 * corresponding elements of [b] within the [tolerance]
 */
fun equals(a: Array<Double>, b: Array<Double>, tolerance: Double = 10e-4): Boolean {
  return a.zip(b).all { equals(it.first, it.second, tolerance = tolerance) }
}

/**
 * Concatenate vertical 1-dim [DenseNDArray]s vertically.
 */
fun concatVectorsV(vararg vectors: DenseNDArray): DenseNDArray {

  require(vectors.all { it.isVector && it.columns == 1 })

  val array = DenseNDArrayFactory.zeros(Shape(vectors.sumBy { it.length }))

  var i = 0

  vectors.forEach {
    (0 until it.length).forEach { j -> array[i++] = it[j] }
  }

  return array
}

/**
 * Compute the similarity of two arrays (already normalized to unit vectors with the [normalize] function).
 * The cosine similarity value is limited in the range [0.0, 1.0] applying a ReLU function.
 *
 * @param a a dense normalized array
 * @param b a dense normalized array
 *
 * @return the similarity of the two arrays
 */
fun similarity(a: DenseNDArray, b: DenseNDArray): Double = maxOf(0.0, a.t.dot(b)[0])

/**
 * Normalize an array to a unit vector with the Euclidean norm.
 *
 * @return this array normalized (can be a copy or not)
 */
fun DenseNDArray.normalize(): DenseNDArray {

  val norm: Double = this.norm2()

  return if (norm != 0.0) this.div(norm) else this
}

/**
 * Return the number in a formatted string with the specified number of digits.
 *
 * @param digits precision specifier
 */
fun Double.format(digits: Int) = java.lang.String.format("%.${digits}f", this)!!
