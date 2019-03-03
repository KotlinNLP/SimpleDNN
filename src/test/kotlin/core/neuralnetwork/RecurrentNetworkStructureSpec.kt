/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.neuralnetwork

import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayer
import com.kotlinnlp.simplednn.core.layers.models.recurrent.simple.SimpleRecurrentLayer
import com.kotlinnlp.simplednn.core.layers.RecurrentStackedLayers
import com.kotlinnlp.simplednn.core.layers.StructureContextWindow
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.nhaarman.mockito_kotlin.mock
import com.nhaarman.mockito_kotlin.whenever
import core.neuralnetwork.utils.RecurrentNetworkStructureUtils
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 *
 */
class RecurrentNetworkStructureSpec : Spek({

  describe("a RecurrentStackedLayers") {

    context("core.layers factory") {

      val contextWindow = mock<StructureContextWindow<DenseNDArray>>()
      val structure = RecurrentNetworkStructureUtils.buildStructure(contextWindow)

      it("should contain an input layer of the expected type") {
        assertTrue { structure.inputLayer is SimpleRecurrentLayer<DenseNDArray> }
      }

      it("should contain an output layer of the expected type") {
        assertTrue { structure.outputLayer is FeedforwardLayer<DenseNDArray> }
      }
    }

    context("without previous and next contexts") {

      val contextWindow = mock<StructureContextWindow<DenseNDArray>>()
      val curStateStructure = RecurrentNetworkStructureUtils.buildStructure(contextWindow)

      whenever(contextWindow.getPrevState()).thenReturn(null as RecurrentStackedLayers<DenseNDArray>?)
      whenever(contextWindow.getNextState()).thenReturn(null as RecurrentStackedLayers<DenseNDArray>?)

      on("focus on the first layer") {

        curStateStructure.curLayerIndex = 0

        it("should return null as previous context") {
          assertNull(curStateStructure.getPrevState())
        }

        it("should return null as next context") {
          assertNull(curStateStructure.getNextState())
        }
      }

      on("focus on the second layer") {

        curStateStructure.curLayerIndex = 1

        it("should return null as previous context") {
          assertNull(curStateStructure.getPrevState())
        }

        it("should return null as next context") {
          assertNull(curStateStructure.getNextState())
        }
      }
    }

    context("with previous context only") {

      val contextWindow = mock<StructureContextWindow<DenseNDArray>>()
      val curStateStructure = RecurrentNetworkStructureUtils.buildStructure(contextWindow)
      val prevStateStructure = RecurrentNetworkStructureUtils.buildStructure(contextWindow)

      whenever(contextWindow.getPrevState()).thenReturn(prevStateStructure)
      whenever(contextWindow.getNextState()).thenReturn(null as RecurrentStackedLayers<DenseNDArray>?)

      on("focus on the first layer") {

        curStateStructure.curLayerIndex = 0

        it("should return the expected layer as previous context") {
          assertEquals(curStateStructure.getPrevState(), prevStateStructure.layers[0])
        }

        it("should return null as next context") {
          assertNull(curStateStructure.getNextState())
        }
      }

      on("focus on the second layer") {

        curStateStructure.curLayerIndex = 1

        it("should return the expected layer as previous context") {
          assertEquals(curStateStructure.getPrevState(), prevStateStructure.layers[1])
        }

        it("should return null as next context") {
          assertNull(curStateStructure.getNextState())
        }
      }
    }

    context("with next context only") {

      val contextWindow = mock<StructureContextWindow<DenseNDArray>>()
      val curStateStructure = RecurrentNetworkStructureUtils.buildStructure(contextWindow)
      val nextStateStructure = RecurrentNetworkStructureUtils.buildStructure(contextWindow)

      whenever(contextWindow.getPrevState()).thenReturn(null as RecurrentStackedLayers<DenseNDArray>?)
      whenever(contextWindow.getNextState()).thenReturn(nextStateStructure)

      on("focus on the first layer") {

        curStateStructure.curLayerIndex = 0

        it("should return null as previous context") {
          assertNull(curStateStructure.getPrevState())
        }

        it("should return the expected layer as next context") {
          assertEquals(curStateStructure.getNextState(), nextStateStructure.layers[0])
        }
      }

      on("focus on the second layer") {

        curStateStructure.curLayerIndex = 1

        it("should return null as previous context") {
          assertNull(curStateStructure.getPrevState())
        }

        it("should return the expected layer as next context") {
          assertEquals(curStateStructure.getNextState(), nextStateStructure.layers[1])
        }
      }
    }

    context("with previous and next contexts") {

      val contextWindow = mock<StructureContextWindow<DenseNDArray>>()
      val curStateStructure = RecurrentNetworkStructureUtils.buildStructure(contextWindow)
      val prevStateStructure = RecurrentNetworkStructureUtils.buildStructure(contextWindow)
      val nextStateStructure = RecurrentNetworkStructureUtils.buildStructure(contextWindow)

      whenever(contextWindow.getPrevState()).thenReturn(prevStateStructure)
      whenever(contextWindow.getNextState()).thenReturn(nextStateStructure)

      on("focus on the first layer") {

        curStateStructure.curLayerIndex = 0

        it("should return the expected layer as previous context") {
          assertEquals(curStateStructure.getPrevState(), prevStateStructure.layers[0])
        }

        it("should return the expected layer as next context") {
          assertEquals(curStateStructure.getNextState(), nextStateStructure.layers[0])
        }
      }

      on("focus on the second layer") {

        curStateStructure.curLayerIndex = 1

        it("should return the expected layer as previous context") {
          assertEquals(curStateStructure.getPrevState(), prevStateStructure.layers[1])
        }

        it("should return the expected layer as next context") {
          assertEquals(curStateStructure.getNextState(), nextStateStructure.layers[1])
        }
      }
    }
  }
})
