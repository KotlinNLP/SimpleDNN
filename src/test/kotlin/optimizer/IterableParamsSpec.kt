/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package optimizer

import com.kotlinnlp.simplednn.simplemath.ndarray.Indices
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.SparseEntry
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArrayFactory
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertTrue

/**
 *
 */
class IterableParamsSpec : Spek({

  describe("an IterableParams") {

    context("Dense params") {

      on("assignValues") {

        val params1 = IterableParamsUtils.buildDenseParams1()
        val params2 = IterableParamsUtils.buildDenseParams2()

        params1.assignValues(params2)

        it("should assign the expected values to the first parameters") {
          assertTrue {
            (params2.unit.weights.values as DenseNDArray)
              .equals(params1.unit.weights.values as DenseNDArray, tolerance = 1.0e-06)
          }
        }

        it("should assign the expected values to the second parameters") {
          assertTrue {
            params2.unit.biases.values.equals(params1.unit.biases.values, tolerance = 1.0e-06)
          }
        }
      }

      on("assignSum") {

        val params1 = IterableParamsUtils.buildDenseParams1()
        val params2 = IterableParamsUtils.buildDenseParams2()

        params1.assignSum(params2)

        it("should assign the expected values to the first parameters") {
          assertTrue {
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.5, 0.3, 0.5),
              doubleArrayOf(-0.8, 0.3, 1.5)
            )).equals(params1.unit.weights.values as DenseNDArray, tolerance = 1.0e-06)
          }
        }

        it("should assign the expected values to the second parameters") {
          assertTrue {
           DenseNDArrayFactory.arrayOf(doubleArrayOf(0.3, 0.0))
             .equals(params1.unit.biases.values, tolerance = 1.0e-06)
          }
        }
      }

      on("assignDiv") {

        val params1 = IterableParamsUtils.buildDenseParams1()

        params1.assignDiv(4)

        it("should assign the expected values to the first parameters") {
          assertTrue {
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.1, 0.2, 0.05),
              doubleArrayOf(0.025, 0.075, 0.225)
            )).equals(params1.unit.weights.values as DenseNDArray, tolerance = 1.0e-06)
          }
        }

        it("should assign the expected values to the second parameters") {
          assertTrue {
           DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.125, 0.025))
             .equals(params1.unit.biases.values, tolerance = 1.0e-06)
          }
        }
      }
    }

    context("Sparse params") {

      on("assignValues") {

        val params1 = IterableParamsUtils.buildSparseParams1()
        val params2 = IterableParamsUtils.buildSparseParams2()

        params1.assignValues(params2)

        it("should assign the expected values to the first parameters") {
          assertTrue {
            (params2.unit.weights.values as SparseNDArray)
              .equals(params1.unit.weights.values as SparseNDArray, tolerance = 1.0e-06)
          }
        }

        it("should assign the expected values to the second parameters") {
          assertTrue {
            params2.unit.biases.values.equals(params1.unit.biases.values, tolerance = 1.0e-06)
          }
        }
      }

      on("assignSum") {

        val params1 = IterableParamsUtils.buildSparseParams1()
        val params2 = IterableParamsUtils.buildSparseParams2()

        params1.assignSum(params2)

        it("should assign the expected values to the first parameters") {
          assertTrue {
            SparseNDArrayFactory.arrayOf(
              activeIndicesValues = arrayOf(
                SparseEntry(Indices(0, 0), 0.4),
                SparseEntry(Indices(1, 1), -0.2),
                SparseEntry(Indices(1, 2), 0.6)
              ),
              shape = Shape(2, 3)
            ).equals(params1.unit.weights.values as SparseNDArray, tolerance = 1.0e-06)
          }
        }

        it("should assign the expected values to the second parameters") {
          assertTrue {
           DenseNDArrayFactory.arrayOf(doubleArrayOf(0.3, 0.0))
             .equals(params1.unit.biases.values, tolerance = 1.0e-06)
          }
        }
      }

      on("assignDiv") {

        val params1 = IterableParamsUtils.buildSparseParams1()

        params1.assignDiv(4)

        it("should assign the expected values to the first parameters") {
          assertTrue {
            SparseNDArrayFactory.arrayOf(
              activeIndicesValues = arrayOf(
                SparseEntry(Indices(0, 0), 0.1),
                SparseEntry(Indices(1, 1), 0.075)
              ),
              shape = Shape(2, 3)
            ).equals(params1.unit.weights.values as SparseNDArray, tolerance = 1.0e-06)
          }
        }

        it("should assign the expected values to the second parameters") {
          assertTrue {
           DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.125, 0.025))
             .equals(params1.unit.biases.values, tolerance = 1.0e-06)
          }
        }
      }
    }
  }
})
