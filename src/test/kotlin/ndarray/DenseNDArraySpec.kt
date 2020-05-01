/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package ndarray

import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.simplemath.concatVectorsV
import com.kotlinnlp.simplednn.simplemath.equals
import com.kotlinnlp.simplednn.simplemath.exp
import com.kotlinnlp.simplednn.simplemath.ndarray.Indices
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArrayMask
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.SparseEntry
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArrayFactory
import com.nhaarman.mockito_kotlin.mock
import com.nhaarman.mockito_kotlin.whenever
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.*

/**
 *
 */
class DenseNDArraySpec : Spek({

  describe("a DenseNDArray") {

    context("class factory methods") {

      context("arrayOf()") {

        val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3, 0.0))

        it("should have the expected length") {
          assertEquals(4, array.length)
        }

        it("should have the expected last index") {
          assertEquals(3, array.lastIndex)
        }

        it("should have the expected number of rows") {
          assertEquals(4, array.rows)
        }

        it("should have the expected number of columns") {
          assertEquals(1, array.columns)
        }

        it("should contain the expected value at index 0") {
          assertEquals(0.1, array[0])
        }

        it("should contain the expected value at index 1") {
          assertEquals(0.2, array[1])
        }

        it("should contain the expected value at index 2") {
          assertEquals(0.3, array[2])
        }

        it("should contain the expected value at index 3") {
          assertEquals(0.0, array[3])
        }
      }

      context("fromRows()") {

        val matrix = DenseNDArrayFactory.fromRows(listOf(
          DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2)),
          DenseNDArrayFactory.arrayOf(doubleArrayOf(0.3, 0.4)),
          DenseNDArrayFactory.arrayOf(doubleArrayOf(0.5, 0.6))
        ))

        it("should have the expected shape") {
          assertEquals(Shape(3, 2), matrix.shape)
        }

        it("should have the expected values") {
          assertTrue {
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.1, 0.2),
              doubleArrayOf(0.3, 0.4),
              doubleArrayOf(0.5, 0.6)
            )).equals(matrix, tolerance = 0.001)
          }
        }

        it("should raise an exception if the shapes are not compatible") {
          assertFailsWith<IllegalArgumentException> {
            DenseNDArrayFactory.fromRows(listOf(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2)),
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.3, 0.4)),
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.7, 0.8, 0.9))
            ))
          }
        }
      }

      context("fromColumns()") {

        val matrix = DenseNDArrayFactory.fromColumns(listOf(
          DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2)),
          DenseNDArrayFactory.arrayOf(doubleArrayOf(0.3, 0.4)),
          DenseNDArrayFactory.arrayOf(doubleArrayOf(0.5, 0.6))
        ))

        it("should have the expected shape") {
          assertEquals(Shape(2, 3), matrix.shape)
        }

        it("should have the expected values") {
          assertTrue {
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.1, 0.3, 0.5),
              doubleArrayOf(0.2, 0.4, 0.6)
            )).equals(matrix, tolerance = 0.001)
          }
        }

        it("should raise an exception if the shapes are not compatible") {
          assertFailsWith<IllegalArgumentException> {
            DenseNDArrayFactory.fromColumns(listOf(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2)),
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.3, 0.4)),
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.7, 0.8, 0.9))
            ))
          }
        }
      }

      context("zeros()") {

        val array = DenseNDArrayFactory.zeros(Shape(2, 3))

        it("should have the expected length") {
          assertEquals(6, array.length)
        }

        it("should have the expected last index") {
          assertEquals(5, array.lastIndex)
        }

        it("should have the expected number of rows") {
          assertEquals(2, array.rows)
        }

        it("should have the expected number of columns") {
          assertEquals(3, array.columns)
        }

        it("should be filled with zeros") {
          (0 until array.length).forEach { assertEquals(0.0, array[it]) }
        }
      }

      context("ones()") {

        val array = DenseNDArrayFactory.ones(Shape(2, 3))

        it("should have the expected length") {
          assertEquals(6, array.length)
        }

        it("should have the expected last index") {
          assertEquals(5, array.lastIndex)
        }

        it("should have the expected number of rows") {
          assertEquals(2, array.rows)
        }

        it("should have the expected number of columns") {
          assertEquals(3, array.columns)
        }

        it("should be filled with ones") {
          (0 until array.length).forEach { assertEquals(1.0, array[it]) }
        }
      }

      context("fill()") {

        val array = DenseNDArrayFactory.fill(shape = Shape(2, 3), value = 0.35)

        it("should have the expected length") {
          assertEquals(6, array.length)
        }

        it("should have the expected last index") {
          assertEquals(5, array.lastIndex)
        }

        it("should have the expected number of rows") {
          assertEquals(2, array.rows)
        }

        it("should have the expected number of columns") {
          assertEquals(3, array.columns)
        }

        it("should be filled with the expected value") {
          (0 until array.length).forEach { assertEquals(0.35, array[it]) }
        }
      }

      context("emptyArray()") {

        val array = DenseNDArrayFactory.emptyArray(Shape(3, 2))

        it("should have the expected length") {
          assertEquals(6, array.length)
        }

        it("should have the expected last index") {
          assertEquals(5, array.lastIndex)
        }

        it("should have the expected number of rows") {
          assertEquals(3, array.rows)
        }

        it("should have the expected number of columns") {
          assertEquals(2, array.columns)
        }
      }

      context("oneHotEncoder()") {

        val array = DenseNDArrayFactory.oneHotEncoder(length = 4, oneAt = 2)

        it("should have the expected length") {
          assertEquals(4, array.length)
        }

        it("should have the expected last index") {
          assertEquals(3, array.lastIndex)
        }

        it("should have the expected length") {
          assertEquals(4, array.length)
        }

        it("should be a column vector") {
          assertEquals(1, array.columns)
        }

        it("should have the expected values") {
          assertTrue { DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 1.0, 0.0)).equals(array) }
        }
      }

      context("random()") {

        val array = DenseNDArrayFactory.random(shape = Shape(216, 648), from = 0.5, to = 0.89)

        it("should have the expected length") {
          assertEquals(139968, array.length)
        }

        it("should have the expected last index") {
          assertEquals(139967, array.lastIndex)
        }

        it("should have the expected number of rows") {
          assertEquals(216, array.rows)
        }

        it("should have the expected number of columns") {
          assertEquals(648, array.columns)
        }

        it("should contain values within the expected range") {
          (0 until array.length).forEach { i ->
            val value = array[i]
            assertTrue { value >= 0.5 && value < 0.89 }
          }
        }
      }

      context("exp()") {

        val power = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.1, 0.2),
          doubleArrayOf(0.3, 0.0)
        ))
        val array = exp(power)

        it("should have the expected length") {
          assertEquals(4, array.length)
        }

        it("should have the expected last index") {
          assertEquals(3, array.lastIndex)
        }

        it("should have the expected number of rows") {
          assertEquals(2, array.rows)
        }

        it("should have the expected number of columns") {
          assertEquals(2, array.columns)
        }

        it("should contain the expected value at index 0") {
          assertTrue { equals(1.105171, array[0, 0], tolerance = 1.0e-06) }
        }

        it("should contain the expected value at index 1") {
          assertTrue { equals(1.221403, array[0, 1], tolerance = 1.0e-06) }
        }

        it("should contain the expected value at index 2") {
          assertTrue { equals(1.349859, array[1, 0], tolerance = 1.0e-06) }
        }

        it("should contain the expected value at index 3") {
          assertTrue { equals(1.0, array[1, 1], tolerance = 1.0e-06) }
        }
      }
    }

    context("equality with tolerance") {

      val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.123, 0.234, 0.345, 0.012))

      context("comparison with different types") {

        val arrayToCompare = SparseNDArrayFactory.arrayOf(
          activeIndicesValues = arrayOf(
            SparseEntry(Indices(0, 0), 0.123),
            SparseEntry(Indices(1, 0), 0.234),
            SparseEntry(Indices(2, 0), 0.345),
            SparseEntry(Indices(3, 0), 0.012)
          ),
          shape = Shape(4))

        it("should return false") {
          assertFalse { array.equals(arrayToCompare, tolerance = 1.0e0-3) }
        }
      }

      context("comparison within the tolerance") {

        val arrayToCompare = DenseNDArrayFactory.arrayOf(
          doubleArrayOf(0.123000001, 0.234000001, 0.345000001, 0.012000001))

        it("should result equal with a large tolerance") {
          assertTrue { array.equals(arrayToCompare, tolerance=1.0e-03) }
        }

        it("should result equal with a strict tolerance") {
          assertTrue { array.equals(arrayToCompare, tolerance=1.0e-08) }
        }
      }

      context("comparison out of tolerance") {

        val arrayToCompare = DenseNDArrayFactory.arrayOf(
          doubleArrayOf(0.12303, 0.23403, 0.34503, 0.01203))

        it("should result not equal") {
          assertFalse { array.equals(arrayToCompare, tolerance=1.0e-05) }
        }
      }
    }

    context("initialization through a double array of 4 elements") {

      val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3, 0.0))

      context("properties") {

        it("should be a vector") {
          assertTrue { array.isVector }
        }

        it("should not be a matrix") {
          assertFalse { array.isMatrix }
        }

        it("should have the expected length") {
          assertEquals(4, array.length)
        }

        it("should have the expected number of rows") {
          assertEquals(4, array.rows)
        }

        it("should have the expected number of columns") {
          assertEquals(1, array.columns)
        }

        it("should have the expected shape") {
          assertEquals(Shape(4), array.shape)
        }
      }

      context("generic methods") {

        it("should be equal to itself") {
          assertTrue { array.equals(array) }
        }

        it("should be equal to its copy") {
          assertTrue { array.equals(array.copy()) }
        }
      }

      context("getRange() method") {

        val a = array.getRange(0, 3)
        val b = array.getRange(2, 4)

        it("should return a range of the expected length") {
          assertEquals(3, a.length)
        }

        it("should return the expected range (0, 3)") {
          assertTrue {
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3)).equals(a)
          }
        }

        it("should return the expected range (2, 4)") {
          assertTrue {
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.3, 0.0)).equals(b)
          }
        }

        it("should raise an IndexOutOfBoundsException requesting for a range out of bounds") {
          assertFailsWith<IndexOutOfBoundsException> {
            array.getRange(2, 6)
          }
        }
      }

      context("transpose") {

        val transposedArray = array.t

        it("should give a transposed array with the expected shape") {
          assertEquals(Shape(1, 4), transposedArray.shape)
        }

        it("should give a transposed array with the expected values") {
          assertEquals(transposedArray[2], 0.3)
        }
      }
    }

    context("isOneHotEncoder() method") {

      val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3, 0.0))
      val oneHotEncoder = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 1.0, 0.0))
      val oneHotEncoderDouble = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 1.0, 1.0, 0.0))
      val oneHotEncoderFake = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.1, 0.0, 0.0))
      val array2 = DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.1, 0.2, 0.3, 0.0),
        doubleArrayOf(0.1, 0.2, 0.3, 0.0)
      ))

      it("should return false on a random array") {
        assertFalse { array.isOneHotEncoder }
      }

      it("should return false on a 2-dim array") {
        assertFalse { array2.isOneHotEncoder }
      }

      it("should return false on an array with one element equal to 0.1") {
        assertFalse { oneHotEncoderFake.isOneHotEncoder }
      }

      it("should return false on an array with two elements equal to 1.0") {
        assertFalse { oneHotEncoderDouble.isOneHotEncoder }
      }

      it("should return true on an array with one element equal to 1.0") {
        assertTrue { oneHotEncoder.isOneHotEncoder }
      }
    }

    context("math methods returning a new NDArray") {

      val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3, 0.0))
      val a = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, 0.3, 0.5, 0.7))
      val n = 0.9

      context("sum(number) method") {

        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(1.0, 1.1, 1.2, 0.9))
        val res = array.sum(n)

        it("should return a new DenseNDArray") {
          assertFalse { array === res }
        }

        it("should return the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }

      context("sum(array) method") {

        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.5, 0.5, 0.8, 0.7))
        val res = array.sum(a)

        it("should return a new DenseNDArray") {
          assertFalse { array === res }
        }

        it("should return the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }

      context("sub(number) method") {

        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.8, -0.7, -0.6, -0.9))
        val res = array.sub(n)

        it("should return a new DenseNDArray") {
          assertFalse { array === res }
        }

        it("should return the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }

      context("sub(array) method") {

        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.3, -0.1, -0.2, -0.7))
        val res = array.sub(a)

        it("should return a new DenseNDArray") {
          assertFalse { array === res }
        }

        it("should assign the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }

      context("reverseSub(number) method") {

        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.8, 0.7, 0.6, 0.9))
        val res = array.reverseSub(n)

        it("should return a new DenseNDArray") {
          assertFalse { array === res }
        }

        it("should return the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }

      context("dot(array) method") {

        val expectedArray = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.04, 0.03, 0.05, 0.07),
          doubleArrayOf(0.08, 0.06, 0.1, 0.14),
          doubleArrayOf(0.12, 0.09, 0.15, 0.21),
          doubleArrayOf(0.0, 0.0, 0.0, 0.0)
        ))
        val res = array.dot(a.t)

        it("should return a new DenseNDArray") {
          assertFalse { array === res }
        }

        it("should throw an error with not compatible shapes") {
          assertFails { array.dot(a) }
        }

        it("should assign the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }

      context("dotLeftMasked(array, mask) method") {

        val a1 = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.5, 0.3),
          doubleArrayOf(1.0, 0.5),
          doubleArrayOf(0.7, 0.6)
        ))
        val a2 = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.2, 0.9),
          doubleArrayOf(0.5, 0.6)
        ))
        val expected = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.1, 0.45),
          doubleArrayOf(0.25, 0.3),
          doubleArrayOf(0.0, 0.0)
        ))
        val res = a1.dotLeftMasked(a2, mask = NDArrayMask(dim1 = intArrayOf(0, 1), dim2 = intArrayOf(0, 1)))

        it("should throw an error with not compatible shapes") {
          val a3 = DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.7, 0.5),
            doubleArrayOf(0.3, 0.2),
            doubleArrayOf(0.3, 0.5),
            doubleArrayOf(0.7, 0.5)
          ))
          assertFails { array.assignDot(a1, a3) }
        }

        it("should assign the expected values") {
          assertTrue { expected.equals(res, tolerance = 1.0e-04) }
        }
      }

      context("dotRightMasked(array, mask) method") {

        val a1 = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.5, 0.3),
          doubleArrayOf(1.0, 0.5),
          doubleArrayOf(0.7, 0.6)
        ))
        val a2 = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.2, 0.9),
          doubleArrayOf(0.5, 0.6)
        ))
        val expected = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.1, 0.18),
          doubleArrayOf(0.2, 0.3),
          doubleArrayOf(0.14, 0.36)
        ))
        val res = a1.dotRightMasked(a2, mask = NDArrayMask(dim1 = intArrayOf(0, 1), dim2 = intArrayOf(0, 1)))

        it("should throw an error with not compatible shapes") {
          val a3 = DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.7, 0.5),
            doubleArrayOf(0.3, 0.2),
            doubleArrayOf(0.3, 0.5),
            doubleArrayOf(0.7, 0.5)
          ))
          assertFails { array.assignDot(a1, a3) }
        }

        it("should assign the expected values") {
          assertTrue { expected.equals(res, tolerance = 1.0e-04) }
        }
      }

      context("prod(number) method") {

        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.09, 0.18, 0.27, 0.0))
        val res = array.prod(n)

        it("should return a new DenseNDArray") {
          assertFalse { array === res }
        }

        it("should return the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }

      context("prod(array) method") {

        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.04, 0.06, 0.15, 0.0))
        val res = array.prod(a)

        it("should return a new DenseNDArray") {
          assertFalse { array === res }
        }

        it("should return the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }

      context("matrix.prod(colVector) method") {

        val matrix = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.1, 0.2, 0.3, 0.0),
          doubleArrayOf(0.4, 0.5, 0.7, 0.9)
        ))
        val colVector = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.2, 0.3))
        val expectedMatrix = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.02, 0.04, 0.06, 0.0),
          doubleArrayOf(0.12, 0.15, 0.21, 0.27)
        ))
        val res = matrix.prod(colVector)

        it("should return a new DenseNDArray") {
          assertFalse { matrix === res }
        }

        it("should return the expected values") {
          assertTrue { res.equals(expectedMatrix, tolerance = 1.0e-04) }
        }
      }

      context("div(number) method") {

        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1111, 0.2222, 0.3333, 0.0))
        val res = array.div(n)

        it("should return a new DenseNDArray") {
          assertFalse { array === res }
        }

        it("should return the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }

      context("div(array) method") {

        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.25, 0.6667, 0.6, 0.0))
        val res = array.div(a)

        it("should return a new DenseNDArray") {
          assertFalse { array === res }
        }

        it("should return the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }

      context("roundInt(threshold) method") {

        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 1.0, 1.0, 0.0))
        val res = array.roundInt(threshold = 0.2)

        it("should return a new DenseNDArray") {
          assertFalse { array === res }
        }

        it("should return the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }

      context("avg() method") {

        it("should return the expected average") {
          assertTrue { equals(0.15, array.avg(), tolerance = 1.0e-08) }
        }
      }

      context("sign() method") {

        val signedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.1, 0.0, 0.7, -0.6))
        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.0, 0.0, 1.0, -1.0))
        val res = signedArray.sign()

        it("should return a new DenseNDArray") {
          assertFalse { signedArray === res }
        }

        it("should return the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }

      context("nonZeroSign() method") {

        val signedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.1, 0.0, 0.7, -0.6))
        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.0, 1.0, 1.0, -1.0))
        val res = signedArray.nonZeroSign()

        it("should return a new DenseNDArray") {
          assertFalse { signedArray === res }
        }

        it("should return the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }

      context("sqrt() method") {

        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.3162, 0.4472, 0.5478, 0.0))
        val res = array.sqrt()

        it("should return a new DenseNDArray") {
          assertFalse { array === res }
        }

        it("should return the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }

      context("pow(number) method") {

        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.2399, 0.3687, 0.4740, 0.0))
        val res = array.pow(0.62)

        it("should return a new DenseNDArray") {
          assertFalse { array === res }
        }

        it("should return the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }

      context("log10() method") {

        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.397940, -0.522879, -0.301030, -0.154902))
        val res = a.log10()

        it("should raise an exception if at least a value is 0.0") {
          assertFailsWith<IllegalArgumentException> { array.log10() }
        }

        it("should return a new DenseNDArray with a valid array") {
          assertFalse { a === res }
        }

        it("should return the expected values with a valid array") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-06) }
        }
      }

      context("ln() method") {

        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.916291, -1.203973, -0.693147, -0.356675))
        val res = a.ln()

        it("should raise an exception if at least a value is 0.0") {
          assertFailsWith<IllegalArgumentException> { array.ln() }
        }

        it("should return a new DenseNDArray with a valid array") {
          assertFalse { a === res }
        }

        it("should return the expected values with a valid array") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-06) }
        }
      }
    }

    context("math methods in-place") {

      val a = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, 0.3, 0.5, 0.7))
      val b = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.7, 0.8, 0.1, 0.4))
      val n = 0.9

      context("assignSum(number) method") {

        val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3, 0.0))
        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(1.0, 1.1, 1.2, 0.9))
        val res = array.assignSum(n)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }

      context("assignSum(array, number) method") {

        val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3, 0.0))
        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(1.3, 1.2, 1.4, 1.6))
        val res = array.assignSum(a, n)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }

      context("assignSum(array, array) method") {

        val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3, 0.0))
        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(1.1, 1.1, 0.6, 1.1))
        val res = array.assignSum(a, b)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }

      context("assignSum(array) method") {

        val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3, 0.0))
        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.5, 0.5, 0.8, 0.7))
        val res = array.assignSum(a)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }

      context("assignSub(number) method") {

        val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3, 0.0))
        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.8, -0.7, -0.6, -0.9))
        val res = array.assignSub(n)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }

      context("assignSub(array) method") {

        val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3, 0.0))
        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.3, -0.1, -0.2, -0.7))
        val res = array.assignSub(a)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }

      context("assignDot(array, array[1-d]) method") {

        val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3, 0.0))
        val a1 = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.28))
        val expectedArray = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.112),
          doubleArrayOf(0.084),
          doubleArrayOf(0.14),
          doubleArrayOf(0.196)
        ))
        val res = array.assignDot(a, a1)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should throw an error with not compatible shapes") {
          assertFails { array.assignDot(a, b.t) }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }

      context("assignDot(array, array[2-d]) method") {

        val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3, 0.0))
        val v = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.3, 0.8))
        val m = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.7, 0.5),
          doubleArrayOf(0.3, 0.2),
          doubleArrayOf(0.3, 0.5),
          doubleArrayOf(0.7, 0.5)
        ))
        val expectedArray = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.61),
          doubleArrayOf(0.25),
          doubleArrayOf(0.49),
          doubleArrayOf(0.61)
        ))
        val res = array.assignDot(m, v)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should throw an error with not compatible shapes") {
          val m2 = DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.7, 0.5),
            doubleArrayOf(0.3, 0.2),
            doubleArrayOf(0.3, 0.5),
            doubleArrayOf(0.7, 0.5)
          ))
          assertFails { array.assignDot(a.t, m2) }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }

      context("assignDotLeftMasked(array[1-d], array[2-d], mask) method") {

        val array = DenseNDArrayFactory.emptyArray(Shape(1, 2))
        val a1 = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.7, 0.3, 0.6)
        ))
        val m = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.5, 0.3),
          doubleArrayOf(1.0, 0.5),
          doubleArrayOf(0.7, 0.6)
        ))
        val expectedArray = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.3, 0.15)
        ))
        val res = array.assignDotLeftMasked(a1, m, aMask = NDArrayMask(dim1 = intArrayOf(0), dim2 = intArrayOf(1)))

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should throw an error with not compatible shapes") {
          val m2 = DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.7, 0.5),
            doubleArrayOf(0.3, 0.2),
            doubleArrayOf(0.3, 0.5),
            doubleArrayOf(0.7, 0.5)
          ))
          assertFails { array.assignDot(a1, m2) }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }

      context("assignDotLeftMasked(array[2-d], array[2-d], mask) method") {

        val array = DenseNDArrayFactory.emptyArray(Shape(3, 2))
        val m1 = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.5, 0.3),
          doubleArrayOf(1.0, 0.5),
          doubleArrayOf(0.7, 0.6)
        ))
        val m2 = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.2, 0.9),
          doubleArrayOf(0.5, 0.6)
        ))
        val expectedArray = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.1, 0.45),
          doubleArrayOf(0.25, 0.3),
          doubleArrayOf(0.0, 0.0)
        ))
        val res = array.assignDotLeftMasked(m1, m2, aMask = NDArrayMask(dim1 = intArrayOf(0, 1), dim2 = intArrayOf(0, 1)))

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should throw an error with not compatible shapes") {
          val m3 = DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.7, 0.5),
            doubleArrayOf(0.3, 0.2),
            doubleArrayOf(0.3, 0.5),
            doubleArrayOf(0.7, 0.5)
          ))
          assertFails { array.assignDot(m1, m3) }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }

      context("assignDotRightMasked(array[1-d], array[2-d], mask) method") {

        val array = DenseNDArrayFactory.emptyArray(Shape(1, 2))
        val a1 = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.7, 0.3, 0.6)
        ))
        val m = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.5, 0.3),
          doubleArrayOf(1.0, 0.5),
          doubleArrayOf(0.7, 0.6)
        ))
        val expectedArray = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.35, 0.15)
        ))
        val mask = NDArrayMask(dim1 = intArrayOf(0, 1), dim2 = intArrayOf(0, 1))
        val res = array.assignDotRightMasked(a1, m, bMask = mask)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should throw an error with not compatible shapes") {
          val m2 = DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.7, 0.5),
            doubleArrayOf(0.3, 0.2),
            doubleArrayOf(0.3, 0.5),
            doubleArrayOf(0.7, 0.5)
          ))
          assertFails { array.assignDot(a1, m2) }
        }

        it("should assign the expected values") {
          assertTrue { expectedArray.equals(array, tolerance = 1.0e-04) }
        }
      }

      context("assignDotRightMasked(array[2-d], array[2-d], mask) method") {

        val array = DenseNDArrayFactory.emptyArray(Shape(3, 2))
        val m1 = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.5, 0.3),
          doubleArrayOf(1.0, 0.5),
          doubleArrayOf(0.7, 0.6)
        ))
        val m2 = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.2, 0.9),
          doubleArrayOf(0.5, 0.6)
        ))
        val expectedArray = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.1, 0.18),
          doubleArrayOf(0.2, 0.3),
          doubleArrayOf(0.14, 0.36)
        ))
        val mask = NDArrayMask(dim1 = intArrayOf(0, 1), dim2 = intArrayOf(0, 1))
        val res = array.assignDotRightMasked(m1, m2, bMask = mask)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should throw an error with not compatible shapes") {
          val m3 = DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.7, 0.5),
            doubleArrayOf(0.3, 0.2),
            doubleArrayOf(0.3, 0.5),
            doubleArrayOf(0.7, 0.5)
          ))
          assertFails { array.assignDot(m1, m3) }
        }

        it("should assign the expected values") {
          assertTrue { expectedArray.equals(array, tolerance = 1.0e-04) }
        }
      }

      context("assignProd(number) method") {

        val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3, 0.0))
        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.09, 0.18, 0.27, 0.0))
        val res = array.assignProd(n)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }

      context("assignProd(array, number) method") {

        val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3, 0.0))
        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.36, 0.27, 0.45, 0.63))
        val res = array.assignProd(a, n)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }

      context("assignProd(array, array) method") {

        val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3, 0.0))
        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.28, 0.24, 0.05, 0.28))
        val res = array.assignProd(a, b)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }

      context("assignProd(array) method") {

        val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3, 0.0))
        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.04, 0.06, 0.15, 0.0))
        val res = array.assignProd(a)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }

      context("assignDiv(number) method") {

        val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3, 0.0))
        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1111, 0.2222, 0.3333, 0.0))
        val res = array.assignDiv(n)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }

      context("assignDiv(array) method") {

        val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3, 0.0))
        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.25, 0.6667, 0.6, 0.0))
        val res = array.assignDiv(a)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }

      context("assignPow(number) method") {

        val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3, 0.0))
        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.2399, 0.3687, 0.4740, 0.0))
        val res = array.assignPow(0.62)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }

      context("assignSqrt(number) method") {

        val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3, 0.0))
        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.3162, 0.4472, 0.5478, 0.0))
        val res = array.assignSqrt()

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }

      context("assignLog10() method") {

        val array1 = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3, 0.0))
        val array2 = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, 0.3, 0.5, 0.7))
        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.397940, -0.522879, -0.301030, -0.154902))
        val res = array2.assignLog10()

        it("should raise an exception if at least a value is 0.0") {
          assertFailsWith<IllegalArgumentException> { array1.assignLog10() }
        }

        it("should return the same DenseNDArray with a valid array") {
          assertTrue { array2 === res }
        }

        it("should assign the expected values with a valid array") {
          assertTrue { array2.equals(expectedArray, tolerance = 1.0e-06) }
        }
      }

      context("assignLn() method") {

        val array1 = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3, 0.0))
        val array2 = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, 0.3, 0.5, 0.7))
        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.916291, -1.203973, -0.693147, -0.356675))
        val res = array2.assignLn()

        it("should raise an exception if at least a value is 0.0") {
          assertFailsWith<IllegalArgumentException> { array1.assignLn() }
        }

        it("should return the same DenseNDArray with a valid array") {
          assertTrue { array2 === res }
        }

        it("should assign the expected values with a valid array") {
          assertTrue { array2.equals(expectedArray, tolerance = 1.0e-06) }
        }
      }

      context("assignRoundInt(threshold) method") {

        val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3, 0.0))
        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 1.0, 1.0, 0.0))
        val res = array.assignRoundInt(threshold = 0.2)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }

      context("randomize(randomGenerator) method") {

        val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3, 0.0))
        val randomGeneratorMock = mock<RandomGenerator>()
        var i = 0
        @Suppress("UNUSED_CHANGED_VALUE")
        whenever(randomGeneratorMock.next()).then { a[i++] } // assign the same values of [a]

        val res = array.randomize(randomGeneratorMock)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should return the expected values") {
          assertTrue { res.equals(a) }
        }
      }
    }

    context("other math methods") {

      val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3, 0.0))

      context("sum() method") {

        it("should give the expected sum of its elements") {
          assertTrue { equals(0.6, array.sum(), tolerance = 1.0e-10) }
        }
      }

      context("norm() method") {

        it("should return the expected norm") {
          assertTrue { equals(0.6, array.norm(), tolerance = 1.0e-05) }
        }
      }

      context("norm2() method") {

        it("should return the expected euclidean norm") {
          assertTrue { equals(0.37417, array.norm2(), tolerance = 1.0e-05) }
        }
      }

      context("argMaxIndex() method") {

        it("should have the expected argmax index") {
          assertEquals(2, array.argMaxIndex())
        }

        it("should have the expected argmax index excluding a given index") {
          assertEquals(1, array.argMaxIndex(exceptIndex = 2))
        }

        it("should have the expected argmax index excluding more indices") {
          assertEquals(0, array.argMaxIndex(exceptIndices = setOf(1, 2)))
        }
      }

      context("max() method") {

        it("should have the expected max value") {
          assertEquals(0.3, array.max())
        }
      }
    }

    context("initialization through an array of 2 double arrays of 4 elements") {

      val array = DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.1, 0.2, 0.3, 0.4),
        doubleArrayOf(0.5, 0.6, 0.7, 0.8)
      ))

      context("properties") {

        it("should not be a vector") {
          assertFalse { array.isVector }
        }

        it("should be a matrix") {
          assertTrue { array.isMatrix }
        }

        it("should have the expected length") {
          assertEquals(8, array.length)
        }

        it("should have the expected number of rows") {
          assertEquals(2, array.rows)
        }

        it("should have the expected number of columns") {
          assertEquals(4, array.columns)
        }

        it("should have the expected shape") {
          assertEquals(Shape(2, 4), array.shape)
        }
      }

      context("generic methods") {

        it("should be equal to itself") {
          assertTrue { array.equals(array) }
        }

        it("should be equal to its copy") {
          assertTrue { array.equals(array.copy()) }
        }
      }

      context("getRange() method") {

        it("should fail the vertical vector require") {
          assertFailsWith<Throwable> {
            array.getRange(2, 4)
          }
        }
      }

      context("getRow() method") {

        val row = array.getRow(1)

        it("should return a row vector") {
          assertEquals(1, row.rows)
        }

        it("should return the expected row values") {
          assertTrue { row.equals(DenseNDArrayFactory.arrayOf(listOf(doubleArrayOf(0.5, 0.6, 0.7, 0.8)))) }
        }
      }

      context("getRows() method") {

        val rows = array.getRows()

        it("should return the expected number of rows") {
          assertEquals(2, rows.size)
        }

        it("should return the expected first row") {
          assertTrue {
            rows[0].equals(DenseNDArrayFactory.arrayOf(listOf(doubleArrayOf(0.1, 0.2, 0.3, 0.4))), tolerance = 0.001)
          }
        }

        it("should return the expected second row") {
          assertTrue {
            rows[1].equals(DenseNDArrayFactory.arrayOf(listOf(doubleArrayOf(0.5, 0.6, 0.7, 0.8))), tolerance = 0.001)
          }
        }
      }

      context("getColumn() method") {

        val column = array.getColumn(1)

        it("should return a column vector") {
          assertEquals(1, column.columns)
        }

        it("should return the expected column values") {
          assertTrue { column.equals(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.2, 0.6))) }
        }
      }

      context("getColumns() method") {

        val columns = array.getColumns()

        it("should return the expected number of columns") {
          assertEquals(4, columns.size)
        }

        it("should return the expected first column") {
          assertTrue {
            columns[0].equals(
              DenseNDArrayFactory.arrayOf(listOf(doubleArrayOf(0.1), doubleArrayOf(0.5))),
              tolerance = 0.001)
          }
        }

        it("should return the expected second column") {
          assertTrue {
            columns[1].equals(
              DenseNDArrayFactory.arrayOf(listOf(doubleArrayOf(0.2), doubleArrayOf(0.6))),
              tolerance = 0.001)
          }
        }

        it("should return the expected third column") {
          assertTrue {
            columns[2].equals(
              DenseNDArrayFactory.arrayOf(listOf(doubleArrayOf(0.3), doubleArrayOf(0.7))),
              tolerance = 0.001)
          }
        }

        it("should return the expected fourth column") {
          assertTrue {
            columns[3].equals(
              DenseNDArrayFactory.arrayOf(listOf(doubleArrayOf(0.4), doubleArrayOf(0.8))),
              tolerance = 0.001)
          }
        }
      }

      context("transpose") {

        val transposedArray = array.t

        it("should give a transposed array with the expected shape") {
          assertEquals(Shape(4, 2), transposedArray.shape)
        }

        it("should give a transposed array with the expected values") {
          assertEquals(transposedArray[2, 1], 0.7)
        }
      }
    }

    context("initialization through zerosLike()") {

      val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3, 0.0)).zerosLike()
      val arrayOfZeros = array.zerosLike()

      it("should have the expected length") {
        assertEquals(array.length, arrayOfZeros.length)
      }

      it("should have the expected values") {
        assertTrue { DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0)).equals(arrayOfZeros) }
      }
    }

    context("initialization through onesLike()") {

      val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3, 0.0)).onesLike()
      val arrayOfOnes = array.onesLike()

      it("should have the expected length") {
        assertEquals(array.length, arrayOfOnes.length)
      }

      it("should have the expected values") {
        assertTrue { DenseNDArrayFactory.arrayOf(doubleArrayOf(1.0, 1.0, 1.0, 1.0)).equals(arrayOfOnes) }
      }
    }

    context("converting a DenseNDArray to zeros") {

      val array = DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.1, 0.2, 0.3, 0.4),
        doubleArrayOf(0.5, 0.6, 0.7, 0.8)
      ))

      context("zeros() method call") {

        array.zeros()

        it("should return an DenseNDArray filled with zeros") {
          (0 until array.length).forEach { i -> assertEquals(0.0, array[i]) }
        }
      }
    }

    context("converting a DenseNDArray to ones") {

      val array = DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.1, 0.2, 0.3, 0.4),
        doubleArrayOf(0.5, 0.6, 0.7, 0.8)
      ))

      context("ones() method call") {

        array.ones()

        it("should return an DenseNDArray filled with ones") {
          (0 until array.length).forEach { i -> assertEquals(1.0, array[i]) }
        }
      }
    }

    context("values assignment") {

      context("assignment through another DenseNDArray") {

        val array = DenseNDArrayFactory.emptyArray(Shape(3, 2))
        val arrayToAssign = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.1, 0.2),
          doubleArrayOf(0.3, 0.4),
          doubleArrayOf(0.5, 0.6)
        ))

        array.assignValues(arrayToAssign)

        it("should contain the expected assigned values") {
          assertTrue { array.equals(arrayToAssign) }
        }
      }

      context("assignment through a number") {

        val array = DenseNDArrayFactory.emptyArray(Shape(3, 2))

        array.assignValues(0.6)

        it("should contain the expected assigned values") {
          (0 until array.length).forEach { i -> assertEquals(0.6, array[i]) }
        }
      }
    }

    context("getters") {

      context("a vertical vector") {
        val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3, 0.0))

        it("should get the correct item") {
          assertEquals(array[2], 0.3)
        }
      }

      context("a horizontal vector") {
        val array = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.1),
          doubleArrayOf(0.2),
          doubleArrayOf(0.3),
          doubleArrayOf(0.0)
        ))

        it("should get the correct item") {
          assertEquals(array[2], 0.3)
        }
      }

      context("a matrix") {
        val array = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.1, 0.2, 0.3),
          doubleArrayOf(0.4, 0.5, 0.6)
        ))

        it("should get the correct item") {
          assertEquals(array[1, 2], 0.6)
        }
      }
    }

    context("setters") {

      context("a vertical vector") {
        val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3, 0.0))

        array[2] = 0.7

        it("should set the correct item") {
          assertEquals(array[2], 0.7)
        }
      }

      context("a horizontal vector") {
        val array = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.1),
          doubleArrayOf(0.2),
          doubleArrayOf(0.3),
          doubleArrayOf(0.0)
        ))

        array[2] = 0.7

        it("should get the correct item") {
          assertEquals(array[2], 0.7)
        }
      }

      context("a matrix") {
        val array = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.1, 0.2, 0.3),
          doubleArrayOf(0.4, 0.5, 0.6)
        ))

        array[1, 2] = 0.7

        it("should get the correct item") {
          assertEquals(array[1, 2], 0.7)
        }
      }
    }

    context("single horizontal concatenation") {

      val array1 = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3))
      val array2 = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, 0.5, 0.6))
      val concatenatedArray = array1.concatH(array2)

      it("should have the expected shape") {
        assertEquals(Shape(3, 2), concatenatedArray.shape)
      }

      it("should have the expected values") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.1, 0.4),
            doubleArrayOf(0.2, 0.5),
            doubleArrayOf(0.3, 0.6))
          ).equals(concatenatedArray)
        }
      }
    }

    context("single vertical concatenation") {

      val array1 = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3))
      val array2 = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, 0.5, 0.6))
      val concatenatedArray = array1.concatV(array2)

      it("should have the expected length") {
        assertEquals(6, concatenatedArray.length)
      }

      it("should have the expected values") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
            .equals(concatenatedArray)
        }
      }
    }

    context("multiple vertical concatenation") {

      val concatenatedArray = concatVectorsV(
        DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3)),
        DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, 0.5, 0.6)),
        DenseNDArrayFactory.arrayOf(doubleArrayOf(0.7, 0.8, 0.9))
      )

      it("should have the expected length") {
        assertEquals(9, concatenatedArray.length)
      }

      it("should have the expected values") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
            .equals(concatenatedArray)
        }
      }
    }

    context("single vertical split") {

      val array1 = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3, 0.4))

      val splitArray: List<DenseNDArray> = array1.splitV(2)

      it("should have the expected length") {
        assertEquals(2, splitArray.size)
      }

      it("should have the expected values") {
        assertEquals(
          listOf(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2)),
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.3, 0.4))
          ),
          splitArray
        )
      }
    }

    context("single vertical split multiple range size") {

      val array1 = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3, 0.4))

      val splitArray: List<DenseNDArray> = array1.splitV(2, 1, 1)

      it("should have the expected length") {
        assertEquals(3, splitArray.size)
      }

      it("should have the expected values") {
        assertEquals(
          listOf(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2)),
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.3)),
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4))
          ),
          splitArray
        )
      }
    }
  }
})
