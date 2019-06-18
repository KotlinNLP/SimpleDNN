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
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.exp

/**
 * Equals within tolerance.
 *
 * @param a a [Double] number
 * @param b a [Double] number
 * @param tolerance it defines the range [[b] - [tolerance], [b] + [tolerance]]
 *
 * @return a [Boolean] which indicates if [a] is equal to [b] within the [tolerance]
 */
fun equals(a: Double, b: Double, tolerance: Double = 1.0e-04): Boolean {

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
fun equals(a: DoubleArray, b: DoubleArray, tolerance: Double = 1.0e-04): Boolean =
  a.zip(b).all { equals(it.first, it.second, tolerance = tolerance) }

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
 * Sum in place the i-th element of [a] to the i-the element of this list
 */
fun List<DenseNDArray>.assignSum(a: List<DenseNDArray>) {

  require(this.size == a.size)

  a.forEachIndexed { i, array ->
    require(this[i].shape == array.shape || (this[i].isVector && array.isVector && this[i].length == array.length))
    this[i].assignSum(array)
  }
}

/**
 * Transform a list of vectors into a matrix.
 */
fun List<DenseNDArray>.toMatrix(): DenseNDArray {

  require(this.all { it.isVector })

  return DenseNDArrayFactory.arrayOf(this.map { it.toDoubleArray() })
}

/**
 * Transform a matrix into a list of vectors.
 */
fun DenseNDArray.toVectors(): List<DenseNDArray> {

  require(this.isMatrix)

  return (0 until this.rows).map { this.getRow(it).t }
}

/**
 * Compute the cosine similarity of two arrays (already normalized with the normalize2 function).
 * The cosine similarity value is limited in the range [0.0, 1.0] applying a ReLU function.
 *
 * @param a a dense normalized array
 * @param b a dense normalized array
 *
 * @return the cosine similarity of the two arrays
 */
fun cosineSimilarity(a: DenseNDArray, b: DenseNDArray): Double = maxOf(0.0, a.t.dot(b)[0])

/**
 * Compute the similarity of two arrays (already normalized to unit vectors with the normalize function) based on the
 * Structural Entropic Distance (SED).
 * The similarity value is limited in the range [0.0, 1.0].
 *
 * References:
 * [Connor R., Moss R. (2012) A Multivariate Correlation Distance for Vector Spaces. In: Navarro G., Pestov V. (eds)
 * Similarity Search and Applications. SISAP 2012. Lecture Notes in Computer Science, vol 7404. Springer, Berlin,
 * Heidelberg](https://doi.org/10.1007/978-3-642-32153-5_15)
 *
 * @param a a dense normalized array
 * @param b a dense normalized array
 *
 * @return the SED similarity of the two arrays
 */
fun sedSimilarity(a: DenseNDArray, b: DenseNDArray): Double {

  require(a.shape == b.shape)

  val exp: Double = (0 until a.length).sumByDouble { i -> complexityExp(a[i], b[i]) }

  return 2.0 - 10.0.pow(exp)
}

/**
 * Build an array with the exponential of the values of a given array.
 *
 * @param a a dense array
 *
 * @return the exponential of the given array
 */
fun exp(a: DenseNDArray): DenseNDArray {

  val ret: DenseNDArray = DenseNDArrayFactory.emptyArray(a.shape)

  (0 until ret.length).forEach { i -> ret[i] = exp(a[i]) }

  return ret
}

/**
 * Simple work-around that make the Math.log() safe for zero or negative values
 *
 * @param value the value
 * @param eps the number to use when the given [value] is less than this
 *
 * @return the logarithm
 */
fun safeLog(value: Double, eps: Double = 1.0e-08): Double = Math.log(if (value >= eps) value else eps)

/**
 * Calculate the SED complexity exponent component of two vectors of the i-th dimension if [a] and [b] are the values of
 * the vectors of the i-th dimension.
 *
 * @param a the i-th element of the first vector
 * @param b the i-th element of the second vector
 *
 * @return the SED complexity exponent component of the i-th dimension of the vectors
 */
private fun complexityExp(a: Double, b: Double): Double =
  0.5 * (negShannonEntropy(a) + negShannonEntropy(b)) - negShannonEntropy(0.5 * (a + b))

/**
 * Apply the negative Shannon's entropy with base 10 (H_10) to a given double number.
 *
 * Shannon's Entropy:
 *  H_a(x) = - x * log_a(x)
 *
 * @param x a double number
 *
 * @return the Shannon's entropy of the given number
 */
private fun negShannonEntropy(x: Double): Double = x * log10(max(1.0e-16, x)) // limited to avoid overflow errors
