/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.arrays

import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.core.arrays.UpdatableSparseArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 *
 */
class UpdatableArraySpec : Spek({

  describe("an UpdatableArray") {

    context("initialization") {

      on("with an NDArray") {

        val updatableArray = UpdatableArray(DenseNDArrayFactory.zeros(Shape(3, 7)))

        it("should contain values with the expected number of rows") {
          assertEquals(3, updatableArray.values.rows)
        }

        it("should contain values with the expected number of columns") {
          assertEquals(7, updatableArray.values.columns)
        }

        it("should raise an Exception when trying to access its structure without setting it") {
          assertFailsWith<UninitializedPropertyAccessException> { updatableArray.updaterSupportStructure }
        }
      }
    }
  }

  describe("an UpdatableDenseArray") {

    context("initialization") {

      on("with the shape") {

        val updatableArray = UpdatableDenseArray(shape = Shape(3, 2))

        it("should contain values with the expected number of rows") {
          assertEquals(3, updatableArray.values.rows)
        }

        it("should contain values with the expected number of columns") {
          assertEquals(2, updatableArray.values.columns)
        }

        it("should contain zeros values") {
          assertTrue {
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.0, 0.0),
              doubleArrayOf(0.0, 0.0),
              doubleArrayOf(0.0, 0.0)
            )).equals(updatableArray.values, tolerance = 1.0e-08)
          }
        }

        it("should raise an Exception when trying to access its structure without setting it") {
          assertFailsWith<UninitializedPropertyAccessException> { updatableArray.updaterSupportStructure }
        }
      }
    }
  }

  describe("an UpdatableSparseArray") {

    context("initialization") {

      on("with the shape") {

        val updatableArray = UpdatableSparseArray(shape = Shape(3, 2))

        it("should contain values with the expected number of rows") {
          assertEquals(3, updatableArray.values.rows)
        }

        it("should contain values with the expected number of columns") {
          assertEquals(2, updatableArray.values.columns)
        }

        it("should contain zeros values") {
          assertEquals(0, updatableArray.values.values.size)
        }

        it("should raise an Exception when trying to access its structure without setting it") {
          assertFailsWith<UninitializedPropertyAccessException> { updatableArray.updaterSupportStructure }
        }
      }
    }
  }
})
