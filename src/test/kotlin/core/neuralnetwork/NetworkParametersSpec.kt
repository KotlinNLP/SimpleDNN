/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.neuralnetwork

import com.kotlinnlp.simplednn.core.layers.LayerInterface
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertEquals
import kotlin.test.assertFalse

/**
 *
 */
class NetworkParametersSpec: Spek({

  describe("a StackedLayersParameters") {

    context("iteration over SimpleRecurrent + Feedforward parameters") {

      val params = StackedLayersParameters(listOf(
        LayerInterface(size = 3),
        LayerInterface(size = 4, connectionType = LayerType.Connection.SimpleRecurrent),
        LayerInterface(size = 2, connectionType = LayerType.Connection.Feedforward)
      ))

      context("iteration 1") {

        val iterator = params.iterator()
        val firstLayerIterator = params.paramsPerLayer[0].iterator()

        it("should return the params of the first iteration of the first layer") {
          assertEquals(iterator.next(), firstLayerIterator.next())
        }
      }

      context("iteration 2") {

        val iterator = params.iterator()
        val firstLayerIterator = params.paramsPerLayer[0].iterator()

        iterator.next()
        firstLayerIterator.next()

        it("should return the params of the second iteration of the first layer") {
          assertEquals(iterator.next(), firstLayerIterator.next())
        }
      }

      context("iteration 3") {

        val iterator = params.iterator()
        val firstLayerIterator = params.paramsPerLayer[0].iterator()

        repeat(2) { iterator.next() }
        repeat(2) { firstLayerIterator.next() }

        it("should return the params of the third iteration of the first layer") {
          assertEquals(iterator.next(), firstLayerIterator.next())
        }
      }

      context("iteration 4") {

        val iterator = params.iterator()
        val secondLayerIterator = params.paramsPerLayer[1].iterator()

        repeat(3) { iterator.next() }

        it("should return the params of the first iteration of the second layer") {
          assertEquals(iterator.next(), secondLayerIterator.next())
        }
      }

      context("iteration 5") {

        val iterator = params.iterator()
        val secondLayerIterator = params.paramsPerLayer[1].iterator()

        repeat(4) { iterator.next() }
        secondLayerIterator.next()

        it("should return the params of the second iteration of the second layer") {
          assertEquals(iterator.next(), secondLayerIterator.next())
        }
      }

      context("iteration 6") {

        val iterator = params.iterator()

        repeat(5) { iterator.next() }

        it("should return false when calling hasNext()") {
          assertFalse { iterator.hasNext() }
        }
      }
    }
  }
})
