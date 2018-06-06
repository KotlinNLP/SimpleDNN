/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package ndarray

import com.kotlinnlp.simplednn.simplemath.equals
import com.kotlinnlp.simplednn.simplemath.ndarray.Indices
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArrayMask
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.SparseEntry
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArrayFactory
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertEquals
import kotlin.test.assertFails
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 *
 */
class SparseNDArraySpec : Spek({

  describe("a SparseNDArray") {

    context("initialization") {

      on("indices out of bounds") {

        it("should raise an Exception") {

          assertFails {
            SparseNDArrayFactory.arrayOf(
              activeIndicesValues = arrayOf(
                Pair(Pair(0, 1), 0.1),
                Pair(Pair(1, 0), 0.5),
                Pair(Pair(1, 3), 0.1),
                Pair(Pair(2, 2), 0.2),
                Pair(Pair(3, 1), 0.3)
              ),
              shape = Shape(4, 3))
          }
        }
      }
    }

    context("iteration") {

      on("2-dim array") {

        val array = SparseNDArrayFactory.arrayOf(
          activeIndicesValues = arrayOf(
            Pair(Pair(0, 1), 0.1),
            Pair(Pair(1, 0), 0.5),
            Pair(Pair(1, 2), 0.1),
            Pair(Pair(2, 2), 0.2),
            Pair(Pair(3, 1), 0.3)
          ),
          shape = Shape(4, 3))
        val iterator = array.iterator()

        it("should return the expected entry on the iteration 1") {
          assertEquals(Pair(Pair(1, 0), 0.5), iterator.next())
        }

        it("should return the expected entry on the iteration 2") {
          assertEquals(Pair(Pair(0, 1), 0.1), iterator.next())
        }

        it("should return the expected entry on the iteration 3") {
          assertEquals(Pair(Pair(3, 1), 0.3), iterator.next())
        }

        it("should return the expected entry on the iteration 4") {
          assertEquals(Pair(Pair(1, 2), 0.1), iterator.next())
        }

        it("should return the expected entry on the iteration 5") {
          assertEquals(Pair(Pair(2, 2), 0.2), iterator.next())
        }

        it("should return false calling hasNext() on the last iteration") {
          assertFalse { iterator.hasNext() }
        }
      }
    }

    context("assignSumMerging()") {

      on("2-dim arrays") {

        val array1 = SparseNDArrayFactory.arrayOf(
          activeIndicesValues = arrayOf(
            Pair(Pair(0, 1), 0.1),
            Pair(Pair(1, 0), 0.5),
            Pair(Pair(1, 2), 0.1),
            Pair(Pair(2, 2), 0.2),
            Pair(Pair(3, 1), 0.3)
          ),
          shape = Shape(4, 3))

        val array2 = SparseNDArrayFactory.arrayOf(
          activeIndicesValues = arrayOf(
            Pair(Pair(0, 1), 0.2),
            Pair(Pair(1, 0), 0.1),
            Pair(Pair(1, 2), 0.1),
            Pair(Pair(2, 2), 0.5),
            Pair(Pair(2, 1), 0.3)
          ),
          shape = Shape(4, 3))

        val expectedArray = SparseNDArrayFactory.arrayOf(
          activeIndicesValues = arrayOf(
            Pair(Pair(0, 1), 0.3),
            Pair(Pair(1, 0), 0.6),
            Pair(Pair(1, 2), 0.2),
            Pair(Pair(2, 1), 0.3),
            Pair(Pair(2, 2), 0.7),
            Pair(Pair(3, 1), 0.3)
          ),
          shape = Shape(4, 3))

        val res = array1.assignSumMerging(array2)

        it("should return the same array") {
          assertTrue(array1 === res)
        }

        it("should contain the expected values") {
          assertTrue(expectedArray.equals(res))
        }
      }
    }

    context("math methods returning a new NDArray") {

      on("dot(denseArray) method") {

        val a1 = SparseNDArrayFactory.arrayOf(
          activeIndicesValues = arrayOf(
            SparseEntry(Indices(0, 0), 0.5),
            SparseEntry(Indices(1, 1), 0.5)
          ),
          shape = Shape(3, 2))

        val a2 = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.2, 0.9),
          doubleArrayOf(0.5, 0.6)
        ))

        val a3 = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.2, 0.9),
          doubleArrayOf(0.5, 0.6),
          doubleArrayOf(0.1, 0.4)
        ))

        val expectedArray = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.1, 0.45),
          doubleArrayOf(0.25, 0.3),
          doubleArrayOf(0.0, 0.0)
        ))

        val res = a1.dot(a2)

        it("should throw an error with not compatible shapes") {
          assertFails { a1.dot(a3) }
        }

        it("should assign the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04) }
        }
      }
    }

    context("math methods in-place") {

      on ("assignValues(denseArray, mask) method") {

        val aS = SparseNDArrayFactory.arrayOf(activeIndicesValues = arrayOf(), shape = Shape(4, 3))
        val aD = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.2, 0.9, 0.4),
          doubleArrayOf(0.5, 0.6, 0.1),
          doubleArrayOf(0.3, 0.4, 0.6),
          doubleArrayOf(0.1, 0.0, 0.1)
        ))
        val bD = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.2, 0.9, 0.4),
          doubleArrayOf(0.5, 0.6, 0.1)
        ))
        val mask = NDArrayMask(dim1 = intArrayOf(0, 1, 1, 3), dim2 = intArrayOf(1, 0, 2, 2))

        val expectedArray = SparseNDArrayFactory.arrayOf(
          activeIndicesValues = arrayOf(
            SparseEntry(Indices(0, 1), 0.9),
            SparseEntry(Indices(1, 0), 0.5),
            SparseEntry(Indices(1, 2), 0.1),
            SparseEntry(Indices(3, 2), 0.1)
          ),
          shape = Shape(4, 3))

        val res = aS.assignValues(aD, mask = mask)

        it("should return the same DenseNDArray") {
          assertTrue { aS === res }
        }

        it("should assign the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04) }
        }

        it("should throw an error with not compatible shapes") {
          assertFails { aS.assignValues(bD, mask = mask) }
        }
      }

      on ("assignDot(sparseArray, denseArray) method") {

        val a = SparseNDArrayFactory.arrayOf(activeIndicesValues = arrayOf(), shape = Shape(4, 3))

        val aS = SparseNDArrayFactory.arrayOf(
          activeIndicesValues = arrayOf(
            SparseEntry(Indices(0, 0), 0.5),
            SparseEntry(Indices(2, 0), 1.0)
          ),
          shape = Shape(4))

        val mS = SparseNDArrayFactory.arrayOf(
          activeIndicesValues = arrayOf(
            SparseEntry(Indices(0, 0), 0.5),
            SparseEntry(Indices(1, 1), 1.0)
          ),
          shape = Shape(4, 2))

        val aD = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.2, 0.9, 0.5)
        ))

        val b = DenseNDArrayFactory.arrayOf(listOf(
          doubleArrayOf(0.2, 0.9, 0.4),
          doubleArrayOf(0.5, 0.6, 0.1)
        ))

        val expectedArray = SparseNDArrayFactory.arrayOf(
          activeIndicesValues = arrayOf(
            SparseEntry(Indices(0, 0), 0.1),
            SparseEntry(Indices(0, 1), 0.45),
            SparseEntry(Indices(0, 2), 0.25),
            SparseEntry(Indices(2, 0), 0.2),
            SparseEntry(Indices(2, 1), 0.9),
            SparseEntry(Indices(2, 2), 0.5)
          ),
          shape = Shape(4, 3))

        val res = a.assignDot(aS, aD)

        it("should return the same DenseNDArray") {
          assertTrue { a === res }
        }

        it("should assign the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04) }
        }

        it("should throw an error with not compatible shapes") {
          assertFails { a.assignDot(aS, b) }
        }

        it("should throw an error with matrices") {
          assertFails { a.assignDot(mS, b) }
        }
      }
    }

    context("other math methods") {

      val array = SparseNDArrayFactory.arrayOf(
        activeIndicesValues = arrayOf(
          Pair(Pair(0, 1), 0.1),
          Pair(Pair(1, 0), 0.5),
          Pair(Pair(1, 2), 0.1),
          Pair(Pair(2, 2), 0.2),
          Pair(Pair(3, 1), 0.3)
        ),
        shape = Shape(4, 3))

      on("sum() method") {

        it("should give the expected sum of its elements") {
          assertEquals(true, equals(1.2, array.sum(), tolerance = 1.0e-10))
        }
      }

      on("max() method") {

        it("should have the expected max value") {
          assertEquals(0.5, array.max())
        }
      }
    }
  }
})
