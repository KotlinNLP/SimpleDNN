/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package deeplearning.birnn

import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNUtils
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import java.util.*
import kotlin.test.assertEquals

/**
 *
 */
class BiRNNUtilsSpec : Spek({

  describe("a BiRNNUtils") {

    val array1 = arrayOf(
      DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.8, 0.8, -1.0, -0.7)),
      DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.7, -0.8, 0.2, -0.7, 0.7)),
      DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.9, 0.9, 0.7, -0.5, 0.5)),
      DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, -0.1, 0.5, -0.2, -0.8)),
      DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.6, 0.6, 0.8, -0.1, -0.3)))

    val array2 = arrayOf(
      DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, -0.6, -1.0, -0.1, -0.4)),
      DenseNDArrayFactory.arrayOf(doubleArrayOf(0.5, -0.9, 0.0, 0.8, 0.3)),
      DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.3, -0.9, 0.3, 1.0, -0.2)),
      DenseNDArrayFactory.arrayOf(doubleArrayOf(0.7, 0.2, 0.3, -0.4, -0.6)),
      DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.2, 0.5, -0.2, -0.9, 0.4)))

    on("concatenate") {

      val result: Array<DenseNDArray> = BiRNNUtils.concatenate(array1, array2)

      val expectedResult = arrayOf(
        DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.8, 0.8, -1.0, -0.7, 0.1, -0.6, -1.0, -0.1, -0.4)),
        DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.7, -0.8, 0.2, -0.7, 0.7, 0.5, -0.9, 0.0, 0.8, 0.3)),
        DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.9, 0.9, 0.7, -0.5, 0.5, -0.3, -0.9, 0.3, 1.0, -0.2)),
        DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, -0.1, 0.5, -0.2, -0.8, 0.7, 0.2, 0.3, -0.4, -0.6)),
        DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.6, 0.6, 0.8, -0.1, -0.3, -0.2, 0.5, -0.2, -0.9, 0.4)))

      it("should return an array of the expected size") {
        assertEquals(expectedResult.size, result.size)
      }

      it("should return an array with elements of the expected shape") {
        assertEquals(true, result.all{ it.shape == Shape(10, 1) })
      }

      it("should return an array with elements of same shape of the expected values") {
        assertEquals(true, expectedResult.zip(result).all{ (a, b) -> a.shape == b.shape })
      }

      it("should return the pre-calculated values") {
        assertEquals(true, Arrays.equals(expectedResult, result))
      }
    }

    on("sumBidirectionalErrors"){
      val result: Array<DenseNDArray> = BiRNNUtils.sumBidirectionalErrors(array1, array2)

      val expectedResult = arrayOf(
        DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.2, 1.3, 0.6, -1.9, -0.3)),
        DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, -0.6, 0.5, -1.1, 0.1)),
        DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.2, 0.0, 1.0, 0.5, 0.3)),
        DenseNDArrayFactory.arrayOf(doubleArrayOf(0.5, -1.0, 0.5, 0.6, -0.5)),
        DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.5, 0.0, -0.2, -0.2, -0.7)))

      it("should return an array of the expected size") {
        assertEquals(expectedResult.size, result.size)
      }

      it("should return an array with elements of the expected shape") {
        assertEquals(true, result.all{ it.shape == Shape(5, 1) })
      }

      it("should return an array with elements of same shape of the expected values") {
        assertEquals(true, expectedResult.zip(result).all{ (a, b) -> a.shape == b.shape })
      }

      it("should return the pre-calculated values") {
        assertEquals(true, Arrays.equals(expectedResult, result))
      }
    }

    on("splitErrors"){

      val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.8, 0.8, -1.0, -0.7, 0.1, -0.6, -1.0, -0.1, -0.4))

      val (result1, result2) = BiRNNUtils.splitErrors(array)

      val expectedResult1 = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.8, 0.8, -1.0, -0.7))
      val expectedResult2 = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, -0.6, -1.0, -0.1, -0.4))

      it("should return the pre-calculated values on ") {
        assertEquals(true, expectedResult1.equals(result1))
        assertEquals(true, expectedResult2.equals(result2))
      }
    }

    on("splitErrorsSequence"){

      val array = arrayOf(
        DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.8, 0.8, -1.0, -0.7, 0.1, -0.6, -1.0, -0.1, -0.4)),
        DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.7, -0.8, 0.2, -0.7, 0.7, 0.5, -0.9, 0.0, 0.8, 0.3)),
        DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.9, 0.9, 0.7, -0.5, 0.5, -0.3, -0.9, 0.3, 1.0, -0.2)),
        DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, -0.1, 0.5, -0.2, -0.8, 0.7, 0.2, 0.3, -0.4, -0.6)),
        DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.6, 0.6, 0.8, -0.1, -0.3, -0.2, 0.5, -0.2, -0.9, 0.4)))

      val (result1, result2) = BiRNNUtils.splitErrorsSequence(array)

      it("should return the pre-calculated values on ") {
        assertEquals(true, Arrays.equals(array1, result1))
        assertEquals(true, Arrays.equals(array2, result2))
      }
    }
  }
})
