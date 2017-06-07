/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package arrays

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.arrays.DistributionArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ELU
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertEquals
import kotlin.test.assertFails
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 *
 */
class AugmentedArraySpec : Spek({

  describe("an AugmentedArray") {

    val initArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.1, 0.01, -0.1, -0.01, 1.0, 10.0, -1.0, -10.0))
    val errors = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, 0.8, 0.1, 0.8, 0.4, 0.1, 0.2, 0.9, 0.2))
    val arrayWrongSize = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.1, 0.01, -0.1))
    val relevance = DistributionArray.uniform(length = 9)
    val relevanceWrongSize = DistributionArray.uniform(length = 4)

    context("initialization") {

      on("with values not assigned") {

        val activableArray = AugmentedArray<DenseNDArray>(size = 9)

        it("should throw an Exception when trying to get errors") {
          assertFailsWith<UninitializedPropertyAccessException> {
            activableArray.errors
          }
        }

        it("should throw an Exception when trying to get relevance") {
          assertFailsWith<UninitializedPropertyAccessException> {
            activableArray.relevance
          }
        }
      }

      on("with an NDArray") {

        val augmentedArray = AugmentedArray(initArray)

        it("should contain values with the expected number of rows") {
          assertEquals(9, augmentedArray.values.rows)
        }

        it("should contain values with the expected number of columns") {
          assertEquals(1, augmentedArray.values.columns)
        }

        it("should contain zeros errors") {
          assertTrue(DenseNDArrayFactory.zeros(initArray.shape).equals(augmentedArray.errors))
        }
      }
    }

    context("errors") {

      on("before assignment") {

        val augmentedArray = AugmentedArray<DenseNDArray>(size = 9)

        it("should throw an Exception when getting errors") {
          assertFailsWith<UninitializedPropertyAccessException> {
            augmentedArray.errors
          }
        }
      }

      on("after assignment, with values initialized") {

        val augmentedArray = AugmentedArray(initArray)

        it("should throw an exception when trying to assign errors with wrong size") {
          assertFails { augmentedArray.assignValues(arrayWrongSize) }
        }

        augmentedArray.assignErrors(errors)

        it("should have the expected assigned errors") {
          assertEquals(true, augmentedArray.errors.equals(errors, tolerance = 1.0e-08))
        }
      }

      on("after assignment, with values not initialized") {

        val augmentedArray = AugmentedArray<DenseNDArray>(size = 9)
        augmentedArray.assignErrors(errors)

        it("should have the expected assigned errors") {
          assertEquals(true, augmentedArray.errors.equals(errors, tolerance = 1.0e-08))
        }
      }
    }

    context("relevance") {

      on("before assignment") {

        val augmentedArray = AugmentedArray<DenseNDArray>(size = 9)

        it("should throw an Exception when getting relevance") {
          assertFailsWith<UninitializedPropertyAccessException> {
            augmentedArray.relevance
          }
        }
      }

      on("after assignment, with values initialized") {

        val augmentedArray = AugmentedArray(initArray)

        it("should throw an exception when trying to assign relevance with wrong size") {
          assertFails { augmentedArray.assignRelevance(relevanceWrongSize) }
        }

        augmentedArray.assignRelevance(relevance)

        it("should have the expected assigned relevance") {
          assertEquals(true, augmentedArray.relevance.values.equals(relevance.values, tolerance = 1.0e-08))
        }
      }

      on("after assignment, with values not initialized") {

        val augmentedArray = AugmentedArray<DenseNDArray>(size = 9)
        augmentedArray.assignRelevance(relevance)

        it("should have the expected assigned errors") {
          assertEquals(true, augmentedArray.relevance.values.equals(relevance.values, tolerance = 1.0e-08))
        }
      }
    }

    context("cloning") {

      on("values not initialized") {

        val activableArray = AugmentedArray<DenseNDArray>(size = 5)
        val cloneArray = activableArray.clone()

        it("should throw an Exception when getting values") {
          assertFailsWith<UninitializedPropertyAccessException> {
            cloneArray.values
          }
        }

        it("should throw an Exception when getting errors") {
          assertFailsWith<UninitializedPropertyAccessException> {
            cloneArray.errors
          }
        }

        it("should throw an Exception when getting relevance") {
          assertFailsWith<UninitializedPropertyAccessException> {
            cloneArray.relevance
          }
        }
      }

      on("values initialized") {

        val augmentedArray = AugmentedArray(initArray)
        augmentedArray.setActivation(ELU())
        augmentedArray.activate()
        augmentedArray.assignErrors(errors)
        augmentedArray.assignRelevance(relevance)

        val cloneArray = augmentedArray.clone()

        it("should have the expected not activated values") {
          assertEquals(true, augmentedArray.valuesNotActivated.equals(cloneArray.valuesNotActivated))
        }

        it("should have the expected activated values") {
          assertEquals(true, augmentedArray.values.equals(cloneArray.values))
        }

        it("should have the expected errors") {
          assertEquals(true, augmentedArray.errors.equals(cloneArray.errors, tolerance = 1.0e-08))
        }

        it("should have the expected relevance") {
          assertEquals(true, augmentedArray.relevance.values.equals(cloneArray.relevance.values, tolerance = 1.0e-08))
        }
      }
    }
  }
})
