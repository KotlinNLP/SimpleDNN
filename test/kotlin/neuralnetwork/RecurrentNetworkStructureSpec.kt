/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package neuralnetwork

import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerStructure
import com.kotlinnlp.simplednn.core.layers.recurrent.simple.SimpleRecurrentLayerStructure
import com.kotlinnlp.simplednn.core.neuralnetwork.structure.recurrent.RecurrentNetworkStructure
import com.kotlinnlp.simplednn.core.neuralnetwork.structure.recurrent.StructureContextWindow
import com.kotlinnlp.simplednn.simplemath.ndarray.DenseNDArray
import com.nhaarman.mockito_kotlin.mock
import com.nhaarman.mockito_kotlin.whenever
import neuralnetwork.utils.RecurrentNetworkStructureUtils
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertEquals

/**
 *
 */
class RecurrentNetworkStructureSpec : Spek({

  describe("a RecurrentNetworkStructure") {

    context("layers factory") {

      val contextWindow = mock<StructureContextWindow<DenseNDArray>>()
      val structure = RecurrentNetworkStructureUtils.buildStructure(contextWindow)

      it("should contain an input layer of the expected type") {
        assertEquals(true, structure.inputLayer is SimpleRecurrentLayerStructure<DenseNDArray>)
      }

      it("should contain an output layer of the expected type") {
        assertEquals(true, structure.outputLayer is FeedforwardLayerStructure<DenseNDArray>)
      }
    }

    context("without previous and next contexts") {

      val contextWindow = mock<StructureContextWindow<DenseNDArray>>()
      val curStateStructure = RecurrentNetworkStructureUtils.buildStructure(contextWindow)

      whenever(contextWindow.getPrevStateStructure()).thenReturn(null as RecurrentNetworkStructure<DenseNDArray>?)
      whenever(contextWindow.getNextStateStructure()).thenReturn(null as RecurrentNetworkStructure<DenseNDArray>?)

      on("focus on the first layer") {

        curStateStructure.curLayerIndex = 0

        it("should return null as previous context") {
          assertEquals(true, curStateStructure.getPrevStateLayer() == null)
        }

        it("should return null as next context") {
          assertEquals(true, curStateStructure.getNextStateLayer() == null)
        }
      }

      on("focus on the second layer") {

        curStateStructure.curLayerIndex = 1

        it("should return null as previous context") {
          assertEquals(true, curStateStructure.getPrevStateLayer() == null)
        }

        it("should return null as next context") {
          assertEquals(true, curStateStructure.getNextStateLayer() == null)
        }
      }
    }

    context("with previous context only") {

      val contextWindow = mock<StructureContextWindow<DenseNDArray>>()
      val curStateStructure = RecurrentNetworkStructureUtils.buildStructure(contextWindow)
      val prevStateStructure = RecurrentNetworkStructureUtils.buildStructure(contextWindow)

      whenever(contextWindow.getPrevStateStructure()).thenReturn(prevStateStructure)
      whenever(contextWindow.getNextStateStructure()).thenReturn(null as RecurrentNetworkStructure<DenseNDArray>?)

      on("focus on the first layer") {

        curStateStructure.curLayerIndex = 0

        it("should return the expected layer as previous context") {
          assertEquals(true, curStateStructure.getPrevStateLayer() == prevStateStructure.layers[0])
        }

        it("should return null as next context") {
          assertEquals(true, curStateStructure.getNextStateLayer() == null)
        }
      }

      on("focus on the second layer") {

        curStateStructure.curLayerIndex = 1

        it("should return the expected layer as previous context") {
          assertEquals(true, curStateStructure.getPrevStateLayer() == prevStateStructure.layers[1])
        }

        it("should return null as next context") {
          assertEquals(true, curStateStructure.getNextStateLayer() == null)
        }
      }
    }

    context("with next context only") {

      val contextWindow = mock<StructureContextWindow<DenseNDArray>>()
      val curStateStructure = RecurrentNetworkStructureUtils.buildStructure(contextWindow)
      val nextStateStructure = RecurrentNetworkStructureUtils.buildStructure(contextWindow)

      whenever(contextWindow.getPrevStateStructure()).thenReturn(null as RecurrentNetworkStructure<DenseNDArray>?)
      whenever(contextWindow.getNextStateStructure()).thenReturn(nextStateStructure)

      on("focus on the first layer") {

        curStateStructure.curLayerIndex = 0

        it("should return null as previous context") {
          assertEquals(true, curStateStructure.getPrevStateLayer() == null)
        }

        it("should return the expected layer as next context") {
          assertEquals(true, curStateStructure.getNextStateLayer() == nextStateStructure.layers[0])
        }
      }

      on("focus on the second layer") {

        curStateStructure.curLayerIndex = 1

        it("should return null as previous context") {
          assertEquals(true, curStateStructure.getPrevStateLayer() == null)
        }

        it("should return the expected layer as next context") {
          assertEquals(true, curStateStructure.getNextStateLayer() == nextStateStructure.layers[1])
        }
      }
    }

    context("with previous and next contexts") {

      val contextWindow = mock<StructureContextWindow<DenseNDArray>>()
      val curStateStructure = RecurrentNetworkStructureUtils.buildStructure(contextWindow)
      val prevStateStructure = RecurrentNetworkStructureUtils.buildStructure(contextWindow)
      val nextStateStructure = RecurrentNetworkStructureUtils.buildStructure(contextWindow)

      whenever(contextWindow.getPrevStateStructure()).thenReturn(prevStateStructure)
      whenever(contextWindow.getNextStateStructure()).thenReturn(nextStateStructure)

      on("focus on the first layer") {

        curStateStructure.curLayerIndex = 0

        it("should return the expected layer as previous context") {
          assertEquals(true, curStateStructure.getPrevStateLayer() == prevStateStructure.layers[0])
        }

        it("should return the expected layer as next context") {
          assertEquals(true, curStateStructure.getNextStateLayer() == nextStateStructure.layers[0])
        }
      }

      on("focus on the second layer") {

        curStateStructure.curLayerIndex = 1

        it("should return the expected layer as previous context") {
          assertEquals(true, curStateStructure.getPrevStateLayer() == prevStateStructure.layers[1])
        }

        it("should return the expected layer as next context") {
          assertEquals(true, curStateStructure.getNextStateLayer() == nextStateStructure.layers[1])
        }
      }
    }
  }
})
