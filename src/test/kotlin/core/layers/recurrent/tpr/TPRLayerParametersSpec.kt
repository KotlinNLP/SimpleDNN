/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.recurrent.tpr

import com.kotlinnlp.simplednn.core.functionalities.initializers.ConstantInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.RandomInitializer
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.core.layers.models.recurrent.tpr.TPRLayerParameters
import com.nhaarman.mockito_kotlin.mock
import com.nhaarman.mockito_kotlin.whenever
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

class TPRLayerParametersSpec: Spek({

  describe("a TPRLayerParameters") {

    context("initialization") {

      context("dense input") {

        var k = 0
        val initValues = doubleArrayOf(
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
            0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
            1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
            1.9, 2.0, 2.1, 2.2, 2.3, 2.4,
            2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
            3.1, 3.2, 3.3, 3.4, 3.5, 3.6,
            3.7, 3.8, 3.9, 4.0, 4.1, 4.2,
            4.3, 4.4, 4.5, 4.6, 4.7, 4.8)
        val randomGenerator = mock<RandomGenerator>()
        whenever(randomGenerator.next()).thenAnswer { initValues[k++] }

        val params = TPRLayerParameters(
            inputSize = 3,
            dRoles = 2,
            dSymbols = 2,
            nRoles = 2,
            nSymbols = 3,
            weightsInitializer = RandomInitializer(randomGenerator),
            biasesInitializer = ConstantInitializer(0.9))

        val wInS = params.wInS.values
        val wInR = params.wInR.values
        val wRecS = params.wRecS.values
        val wRecR = params.wRecR.values
        val bS = params.bS.values
        val bR = params.bR.values
        val S = params.S.values
        val R = params.R.values

        it("should contain the expected initialized weights of the input -> Symbols matrix") {
          (0 until wInS.length).forEach { i -> assertEquals(initValues[i], wInS[i]) }
        }

        it("should contain the expected initialized weights of the input -> Roles matrix") {
          (0 until wInR.length).forEach { i -> assertEquals(initValues[i + 9], wInR[i]) }
        }

        it("should contain the expected initialized weights of the recurrent -> Symbols matrix") {
          (0 until wRecS.length).forEach { i -> assertEquals(initValues[i + 15], wRecS[i]) }
        }

        it("should contain the expected initialized weights of the recurrent -> Roles matrix") {
          (0 until wRecR.length).forEach { i -> assertEquals(initValues[i + 27], wRecR[i]) }
        }

        it("should contain the expected initialized biases of Symbols") {
          (0 until bS.length).forEach { i -> assertEquals(0.9, bS[i]) }
        }

        it("should contain the expected initialized biases of Roles") {
          (0 until bR.length).forEach { i -> assertEquals(0.9, bR[i]) }
        }

        it("should contain the expected initialized weights of the Symbols embeddings matrix") {
          (0 until S.length).forEach { i -> assertEquals(initValues[i + 35], S[i]) }
        }

        it("should contain the expected initialized weights of the Role embeddings matrix") {
          (0 until R.length).forEach { i -> assertEquals(initValues[i + 41], R[i]) }
        }
      }

      // TODO: reintroduce tests for sparse input
    }

    context("iteration") {

      val params = TPRLayerParameters(
          inputSize = 3,
          dRoles = 2,
          dSymbols = 2,
          nRoles = 2,
          nSymbols = 3)

      val iterator = params.iterator()

      val wInS = params.wInS
      val wInR = params.wInR
      val wRecS = params.wRecS
      val wRecR = params.wRecR
      val bS = params.bS
      val bR = params.bR
      val S = params.S
      val R = params.R

      context("iteration 1") {
        it("should return the weights of the input -> Symbols matrix") {
          assertEquals(wInS, iterator.next())
        }
      }

      context("iteration 2") {
        it("should return the weights of the input -> Roles matrix") {
          assertEquals(wInR, iterator.next())
        }
      }

      context("iteration 3") {
        it("should return the weights of the recurrent -> Symbols matrix") {
          assertEquals(wRecS, iterator.next())
        }
      }

      context("iteration 4") {
        it("should return the weights of the recurrent -> Roles matrix") {
          assertEquals(wRecR, iterator.next())
        }
      }

      context("iteration 5") {
        it("should return the weights of the Symbols embeddings matrix") {
          assertEquals(S, iterator.next())
        }
      }

      context("iteration 6") {
        it("should return the weights of the Role embeddings matrix") {
          assertEquals(R, iterator.next())
        }
      }

      context("iteration 7") {
        it("should return the biases of the input gate") {
          assertEquals(bS, iterator.next())
        }
      }

      context("iteration 8") {
        it("should return the biases of the output gate") {
          assertEquals(bR, iterator.next())
        }
      }
    }

    context("copy") {
      val params = TPRLayerParameters(
          inputSize = 3,
          dRoles = 2,
          dSymbols = 2,
          nRoles = 2,
          nSymbols = 3)

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