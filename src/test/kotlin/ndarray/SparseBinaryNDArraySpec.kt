/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package ndarray

import com.kotlinnlp.simplednn.simplemath.ndarray.Indices
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertEquals
import kotlin.test.assertFalse

/**
 *
 */
class SparseBinaryNDArraySpec : Spek({

  describe("a SparseBinaryNDArray") {

    context("iteration") {

      context("row vector") {

        val array = SparseBinaryNDArrayFactory.arrayOf(activeIndices = listOf(1, 3, 19), shape = Shape(1, 20))
        val iterator = array.iterator()

        it("should return the expected first indices Pair") {
          assertEquals(Pair(0, 1), iterator.next())
        }

        it("should return the expected second indices Pair") {
          assertEquals(Pair(0, 3), iterator.next())
        }

        it("should return the expected third indices Pair") {
          assertEquals(Pair(0, 19), iterator.next())
        }

        it("should return false calling hasNext() at the fourth iteration") {
          assertFalse { iterator.hasNext() }
        }
      }

      context("column vector") {

        val array = SparseBinaryNDArrayFactory.arrayOf(activeIndices = listOf(1, 3, 19), shape = Shape(20))
        val iterator = array.iterator()

        it("should return the expected first indices Pair") {
          assertEquals(Pair(1, 0), iterator.next())
        }

        it("should return the expected second indices Pair") {
          assertEquals(Pair(3, 0), iterator.next())
        }

        it("should return the expected third indices Pair") {
          assertEquals(Pair(19, 0), iterator.next())
        }

        it("should return false calling hasNext() at the fourth iteration") {
          assertFalse { iterator.hasNext() }
        }
      }

      context("a 2-dim array") {

        val array = SparseBinaryNDArrayFactory.arrayOf(
          activeIndices = arrayOf(Indices(1, 0), Indices(5, 19), Indices(12, 6)),
          shape = Shape(15, 20))
        val iterator = array.iterator()

        it("should return the expected first indices Pair") {
          assertEquals(Pair(1, 0), iterator.next())
        }

        it("should return the expected second indices Pair") {
          assertEquals(Pair(12, 6), iterator.next())
        }

        it("should return the expected third indices Pair") {
          assertEquals(Pair(5, 19), iterator.next())
        }

        it("should return false calling hasNext() at the fourth iteration") {
          assertFalse { iterator.hasNext() }
        }
      }
    }
  }
})
