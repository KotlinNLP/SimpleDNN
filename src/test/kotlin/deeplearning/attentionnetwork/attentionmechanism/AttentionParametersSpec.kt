/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package deeplearning.attentionnetwork.attentionmechanism

import com.kotlinnlp.simplednn.core.functionalities.initializers.RandomInitializer
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.attentionmechanism.AttentionParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.nhaarman.mockito_kotlin.mock
import com.nhaarman.mockito_kotlin.whenever
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import kotlin.test.assertTrue

/**
 *
 */
class AttentionParametersSpec : Spek({

  describe("an AttentionParameters") {

    context("initialization") {

      val randomGenerator = mock<RandomGenerator>()
      var i = 0.0
      whenever(randomGenerator.next()).thenAnswer { i++ }

      val params = AttentionParameters(attentionSize = 2, initializer = RandomInitializer(randomGenerator))

      it("should have a context vector with the expected initialized values") {
        assertTrue {
          params.contextVector.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 1.0)),
            tolerance = 1.0e-06
          )
        }
      }
    }
  }
})
