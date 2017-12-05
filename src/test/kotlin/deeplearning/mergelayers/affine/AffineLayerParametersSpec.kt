/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package deeplearning.mergelayers.affine

import com.kotlinnlp.simplednn.core.functionalities.initializers.FixedValueInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.RandomInitializer
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.deeplearning.mergelayers.affine.AffineLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray
import com.nhaarman.mockito_kotlin.mock
import com.nhaarman.mockito_kotlin.whenever
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.*
import kotlin.test.assertEquals
import kotlin.test.assertFails
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 *
 */
class AffineLayerParametersSpec : Spek({

  describe("a AffineLayerParametersS") {

    context("initialization") {

      on("dense input") {

        var k = 0
        val initValues = doubleArrayOf(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        val randomGenerator = mock<RandomGenerator>()
        whenever(randomGenerator.next()).thenAnswer { initValues[k++] }

        val params = AffineLayerParameters(
          inputSize1 = 2,
          inputSize2 = 3,
          outputSize = 2,
          weightsInitializer = RandomInitializer(randomGenerator),
          biasesInitializer = FixedValueInitializer(0.9))

        val w1 = params.w1.values
        val w2 = params.w2.values
        val b = params.b.values

        it("should contain a dense w1") {
          assertTrue { w1 is DenseNDArray }
        }

        it("should contain a dense w2") {
          assertTrue { w2 is DenseNDArray }
        }

        it("should contain the expected initialized w1") {
          (0 until w1.length).forEach { i -> assertEquals(initValues[i], w1[i]) }
        }

        it("should contain the expected initialized w2") {
          (0 until w2.length).forEach { i -> assertEquals(initValues[4 + i], w2[i]) }
        }

        it("should contain the expected initialized biases") {
          (0 until b.length).forEach { i -> assertEquals(0.9, b[i]) }
        }
      }

      on("sparse input") {

        val params = AffineLayerParameters(
          inputSize1 = 2,
          inputSize2 = 3,
          outputSize = 2,
          sparseInput = true,
          weightsInitializer = null,
          biasesInitializer = null)

        val w1 = params.w1.values
        val w2 = params.w2.values

        it("should contain a sparse w1") {
          assertTrue { w1 is SparseNDArray }
        }

        it("should contain a sparse w2") {
          assertTrue { w2 is SparseNDArray }
        }

        it("should throw an Exception when trying to initialize them") {
          assertFails {
            AffineLayerParameters(
              inputSize1 = 2,
              inputSize2 = 3,
              outputSize = 2,
              sparseInput = true,
              weightsInitializer = FixedValueInitializer(0.1),
              biasesInitializer = FixedValueInitializer(0.1))
          }
        }
      }
    }

    context("iteration") {

      val params = AffineLayerParameters(inputSize1 = 2, inputSize2 = 3, outputSize = 2)

      val iterator = params.iterator()

      val w1 = params.w1
      val w2 = params.w2
      val b = params.b

      on("iteration 1") {
        it("should return w1") {
          assertEquals(w1, iterator.next())
        }
      }

      on("iteration 2") {
        it("should return w2") {
          assertEquals(w2, iterator.next())
        }
      }

      on("iteration 3") {
        it("should return the biases") {
          assertEquals(b, iterator.next())
        }
      }

      on("iteration 4") {
        it("should return true when calling hasNext()") {
          assertFalse(iterator.hasNext())
        }
      }
    }
  }
})
