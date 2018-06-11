/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.recurrent.simple

import com.kotlinnlp.simplednn.core.functionalities.initializers.ConstantInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.RandomInitializer
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.core.layers.models.recurrent.simple.SimpleRecurrentLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray
import com.nhaarman.mockito_kotlin.mock
import com.nhaarman.mockito_kotlin.whenever
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.*

/**
 *
 */
class SimpleRecurrentLayerParametersSpec : Spek({

  describe("a SimpleRecurrentLayerParameters") {

    context("initialization") {

      on("dense input") {

        var k = 0
        val initValues = doubleArrayOf(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        val randomGenerator = mock<RandomGenerator>()
        whenever(randomGenerator.next()).thenAnswer { initValues[k++] }

        val params = SimpleRecurrentLayerParameters(
          inputSize = 3,
          outputSize = 2,
          weightsInitializer = RandomInitializer(randomGenerator),
          biasesInitializer = ConstantInitializer(0.9))

        val w = params.unit.weights.values
        val b = params.unit.biases.values
        val wr = params.unit.recurrentWeights.values

        it("should contain the expected initialized weights") {
          (0 until w.length).forEach { i -> assertEquals(initValues[i], w[i]) }
        }

        it("should contain the expected initialized biases") {
          (0 until b.length).forEach { i -> assertEquals(0.9, b[i]) }
        }

        it("should contain the expected initialized recurrent weights") {
          (0 until wr.length).forEach { i -> assertEquals(initValues[i + 6], wr[i]) }
        }
      }

      on("sparse input") {

        val params = SimpleRecurrentLayerParameters(
          inputSize = 3,
          outputSize = 2,
          sparseInput = true,
          weightsInitializer = null,
          biasesInitializer = null)

        val w = params.unit.weights.values

        it("should contain sparse weights") {
          assertTrue { w is SparseNDArray }
        }

        it("should throw an Exception when trying to initialize") {
          assertFails {
            SimpleRecurrentLayerParameters(
              inputSize = 3,
              outputSize = 2,
              sparseInput = true,
              weightsInitializer = ConstantInitializer(0.1),
              biasesInitializer = ConstantInitializer(0.1))
          }
        }
      }
    }

    context("iteration") {

      val params = SimpleRecurrentLayerParameters(inputSize = 3, outputSize = 2)

      val iterator = params.iterator()

      val w = params.unit.weights
      val b = params.unit.biases
      val wr = params.unit.recurrentWeights

      on("iteration 1") {
        it("should return the weights") {
          assertEquals(w, iterator.next())
        }
      }

      on("iteration 2") {
        it("should return the biases") {
          assertEquals(b, iterator.next())
        }
      }

      on("iteration 3") {
        it("should return the recurrent weights") {
          assertEquals(wr, iterator.next())
        }
      }

      on("iteration 4") {
        it("should return true when calling hasNext()") {
          assertFalse(iterator.hasNext())
        }
      }
    }

    context("copy") {

      val params = SimpleRecurrentLayerParameters(inputSize = 3, outputSize = 2)
      val clonedParams = params.copy()

      it("should return a new element") {
        assertNotEquals(params, clonedParams)
      }

      it("should return params with same values") {
        assertTrue {
          params.zip(clonedParams).all { it.first.values.equals(it.second.values, tolerance = 1.0e-06) }
        }
      }
    }
  }
})
