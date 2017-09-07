/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package ndarray

import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals

/**
 *
 */
class ShapeSpec : Spek({

  describe("a Shape") {

    context("shape of dimensions (1, 1)") {

      val shape = Shape(1, 1)
      val inverted = shape.inverse

      on("equals") {

        it("should return true if called passing a Shape with same dimensions") {
          assertEquals(Shape(1, 1), shape)
        }

        it("should return true if called passing a Shape with inverted dimensions") {
          assertEquals(inverted, shape)
        }

        it("should return true if called passing its inverse") {
          assertEquals(inverted, shape)
        }

        it("should return false if called passing a Shape with different dimensions") {
          assertNotEquals(Shape(5), shape)
        }
      }

      on("inverted") {

        it("should have the expected dim 1") {
          assertEquals(inverted.dim1, 1)
        }

        it("should have the expected dim 2") {
          assertEquals(inverted.dim2, 1)
        }
      }
    }

    context("vertical shape of length 4") {

      val shape = Shape(4)
      val inverted = shape.inverse

      on("equals") {

        it("should return true if called passing a Shape with same dimensions") {
          assertEquals(Shape(4), shape)
        }

        it("should return false if called passing a Shape with inverted dimensions") {
          assertNotEquals(Shape(1, 4), shape)
        }

        it("should return false if called passing its inverse") {
          assertNotEquals(inverted, shape)
        }

        it("should return false if called passing a Shape with different dimensions") {
          assertNotEquals(Shape(6), shape)
        }
      }

      on("inverted") {

        it("should return a horizontal shape") {
          assertEquals(inverted.dim1, 1)
        }

        it("should have length 4") {
          assertEquals(inverted.dim2, 4)
        }
      }
    }

    context("horizontal shape of length 4") {

      val shape = Shape(1, 4)
      val inverted = shape.inverse

      on("equals") {

        it("should return true if called passing a Shape with same dimensions") {
          assertEquals(Shape(1, 4), shape)
        }

        it("should return false if called passing a Shape with inverted dimensions") {
          assertNotEquals(Shape(4), shape)
        }

        it("should return false if called passing its inverse") {
          assertNotEquals(inverted, shape)
        }

        it("should return false if called passing a Shape with different dimensions") {
          assertNotEquals(Shape(6), shape)
        }
      }

      on("inverted") {

        it("should return a vertical shape") {
          assertEquals(inverted.dim2, 1)
        }

        it("should have length 4") {
          assertEquals(inverted.dim1, 4)
        }
      }
    }

    context("bi-dimensional shape") {

      val shape = Shape(3, 4)
      val inverted = shape.inverse

      on("equals") {

        it("should return true if called passing a Shape with same dimensions") {
          assertEquals(Shape(3, 4), shape)
        }

        it("should return false if called passing a Shape with inverted dimensions") {
          assertNotEquals(Shape(4, 3), shape)
        }

        it("should return false if called passing its inverse") {
          assertNotEquals(inverted, shape)
        }

        it("should return false if called passing a Shape with different dimensions") {
          assertNotEquals(Shape(4, 6), shape)
        }
      }

      on("inverted") {

        it("should have the expected dim 1") {
          assertEquals(inverted.dim1, 4)
        }

        it("should have the expected dim 2") {
          assertEquals(inverted.dim2, 3)
        }
      }
    }
  }
})
